# ============================================================
# OOC Fmax Binary Search Script for Vivado
# Assumes this .tcl file lives in the same folder as the RTL
# ============================================================

# ============================================
# User settings
# ============================================
set top        "layernorm"
set part       "xc7z015clg485-3"
set clk_port   "clk"

# Add all RTL files needed by this module
set rtl_files [list \
    "layernorm.sv" \
    "linear.sv" \
    "fixed_isqrt.sv" \
    "conv1d_parallel.sv" \
    "gelu_lut_wrapper.sv" \
    "gelu_lut.sv" \
]

set build_dir  "build_ooc_fmax"

# Search bounds in ns
set low_pass_ns  300.0
set high_fail_ns 10.0

# Stop when interval is below this
set tolerance_ns 0.02

# ============================================
# Resolve script directory and move there
# ============================================
set script_dir [file dirname [file normalize [info script]]]
cd $script_dir
puts "Running from script directory: $script_dir"

# ============================================
# Build absolute RTL file list and check files
# ============================================
set abs_rtl_files [list]
foreach f $rtl_files {
    set abs_f [file join $script_dir $f]
    puts "Checking RTL file: $abs_f"
    if {![file exists $abs_f]} {
        error "RTL file not found: $abs_f"
    }
    lappend abs_rtl_files $abs_f
}

# ============================================
# Helper proc: run one OOC implementation
# ============================================
proc run_one_ooc {top part clk_port rtl_files period_ns run_dir} {
    puts ""
    puts "=================================================="
    puts "Running OOC build for period = $period_ns ns"
    puts "=================================================="

    file mkdir $run_dir

    # Clean current in-memory design if one exists
    catch {close_design}
    catch {close_project}

    create_project -in_memory ooc_tmp -part $part

    # Read RTL
    foreach f $rtl_files {
        puts "Reading RTL: $f"
        read_verilog -sv $f
    }

    # OOC synthesis first
    synth_design -top $top -part $part -mode out_of_context

    # Now the synthesized design is open, so get_ports works
    create_clock -name core_clk -period $period_ns [get_ports $clk_port]

    # Implementation
    opt_design
    place_design
    phys_opt_design
    route_design

    # Reports
    report_timing_summary -file [file join $run_dir "timing.rpt"]
    report_utilization    -file [file join $run_dir "util.rpt"]
    report_timing -max_paths 10 -path_type summary -file [file join $run_dir "timing_top10.rpt"]
    report_timing -max_paths 3 -path_type full_clock_expanded -input_pins -file [file join $run_dir "timing_detailed.rpt"]
    write_checkpoint -force [file join $run_dir "${top}_routed.dcp"]

    # Extract worst setup slack
    set paths [get_timing_paths -delay_type max -max_paths 1]
    if {[llength $paths] == 0} {
        puts "WARNING: No timing paths found."
        return [list 0 -9999.0]
    }

    set wns [get_property SLACK [lindex $paths 0]]
    puts "WNS = $wns ns"

    if {$wns >= 0.0} {
        return [list 1 $wns]
    } else {
        return [list 0 $wns]
    }
}

# ============================================
# Main search flow
# ============================================
file mkdir $build_dir

# First verify lower bound passes
set res_low [run_one_ooc $top $part $clk_port $abs_rtl_files $low_pass_ns [file join $build_dir "bound_low"]]
set pass_low [lindex $res_low 0]
set wns_low  [lindex $res_low 1]

if {!$pass_low} {
    error "Lower bound ${low_pass_ns} ns FAILED (WNS=$wns_low). Increase low_pass_ns."
}

# Then verify upper bound fails
set res_high [run_one_ooc $top $part $clk_port $abs_rtl_files $high_fail_ns [file join $build_dir "bound_high"]]
set pass_high [lindex $res_high 0]
set wns_high  [lindex $res_high 1]

if {$pass_high} {
    error "Upper bound ${high_fail_ns} ns PASSED (WNS=$wns_high). Decrease high_fail_ns."
}

set best_pass_ns  $low_pass_ns
set best_pass_wns $wns_low

while {[expr {abs($low_pass_ns - $high_fail_ns)}] > $tolerance_ns} {
    set mid_ns [expr {($low_pass_ns + $high_fail_ns) / 2.0}]
    set run_tag [format "try_%0.3fns" $mid_ns]
    regsub -all {\.} $run_tag "_" run_tag

    set res [run_one_ooc $top $part $clk_port $abs_rtl_files $mid_ns [file join $build_dir $run_tag]]
    set pass [lindex $res 0]
    set wns  [lindex $res 1]

    if {$pass} {
        set best_pass_ns  $mid_ns
        set best_pass_wns $wns
        set low_pass_ns   $mid_ns
    } else {
        set high_fail_ns $mid_ns
    }

    puts "Current best passing period = $best_pass_ns ns"
}

set fmax_mhz [expr {1000.0 / $best_pass_ns}]

puts ""
puts "=================================================="
puts "FINAL RESULT"
puts "Top module           : $top"
puts "Part                 : $part"
puts "Clock port           : $clk_port"
puts "Best passing period  : $best_pass_ns ns"
puts "Best passing WNS     : $best_pass_wns ns"
puts "Estimated OOC Fmax   : $fmax_mhz MHz"
puts "Results directory    : [file join $script_dir $build_dir]"
puts "=================================================="