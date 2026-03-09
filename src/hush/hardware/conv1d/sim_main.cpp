#include "Vconv1d_tb_verilator.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    Vconv1d_tb_verilator* top = new Vconv1d_tb_verilator;

    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("conv1d_tb_verilator.vcd");

    vluint64_t time_counter = 0;
    while (!Verilated::gotFinish()) {
        top->eval();
        tfp->dump(time_counter);
        time_counter++;
    }

    tfp->close();
    delete top;
    delete tfp;
    return 0;
}
