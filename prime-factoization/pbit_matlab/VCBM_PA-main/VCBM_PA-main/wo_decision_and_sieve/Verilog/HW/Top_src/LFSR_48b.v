//////////////////////////////////////////////////////////////////////////////////
// Company: Korea University
// Engineer: Hyunjin Kim & Hyundo Jung
// 
// Create Date: 2022/09/19 11:00:27
// Design Name: Probabilistic Prime Factorization Machine
// Module Name: LFSR_48b
// Project Name: VCBM_PA
// Target Devices: Xilinx Artix-7 (Zynq-7000)
// Tool Versions: Vivado 2020.2
// Description: 48-bit length LFSR that generates 16-bit random numbers per period for p-bit.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module LFSR_48b(
    input clk, rst, operation_start,
    input [48-1:0] seed,
    output [16-1:0] LFSR_out
    );
    
    reg [64-1:0] LFSR_bit;
    
    // sixteen taps for generating next 16 random bits of 48-bit LFSR
    // referenced https://docs.xilinx.com/v/u/en-US/xapp052
    always @(*) begin 
        LFSR_bit[15] <= ~(LFSR_bit[63] ^ LFSR_bit[62] ^ LFSR_bit[36] ^ LFSR_bit[35]);
        LFSR_bit[14] <= ~(LFSR_bit[62] ^ LFSR_bit[61] ^ LFSR_bit[35] ^ LFSR_bit[34]);
        LFSR_bit[13] <= ~(LFSR_bit[61] ^ LFSR_bit[60] ^ LFSR_bit[34] ^ LFSR_bit[33]);
        LFSR_bit[12] <= ~(LFSR_bit[60] ^ LFSR_bit[59] ^ LFSR_bit[33] ^ LFSR_bit[32]);
        LFSR_bit[11] <= ~(LFSR_bit[59] ^ LFSR_bit[58] ^ LFSR_bit[32] ^ LFSR_bit[31]);
        LFSR_bit[10] <= ~(LFSR_bit[58] ^ LFSR_bit[57] ^ LFSR_bit[31] ^ LFSR_bit[30]);
        LFSR_bit[9] <= ~(LFSR_bit[57] ^ LFSR_bit[56] ^ LFSR_bit[30] ^ LFSR_bit[29]);
        LFSR_bit[8] <= ~(LFSR_bit[56] ^ LFSR_bit[55] ^ LFSR_bit[29] ^ LFSR_bit[28]);
        LFSR_bit[7] <= ~(LFSR_bit[55] ^ LFSR_bit[54] ^ LFSR_bit[28] ^ LFSR_bit[27]);
        LFSR_bit[6] <= ~(LFSR_bit[54] ^ LFSR_bit[53] ^ LFSR_bit[27] ^ LFSR_bit[26]);
        LFSR_bit[5] <= ~(LFSR_bit[53] ^ LFSR_bit[52] ^ LFSR_bit[26] ^ LFSR_bit[25]);
        LFSR_bit[4] <= ~(LFSR_bit[52] ^ LFSR_bit[51] ^ LFSR_bit[25] ^ LFSR_bit[24]);
        LFSR_bit[3] <= ~(LFSR_bit[51] ^ LFSR_bit[50] ^ LFSR_bit[24] ^ LFSR_bit[23]);
        LFSR_bit[2] <= ~(LFSR_bit[50] ^ LFSR_bit[49] ^ LFSR_bit[23] ^ LFSR_bit[22]);
        LFSR_bit[1] <= ~(LFSR_bit[49] ^ LFSR_bit[48] ^ LFSR_bit[22] ^ LFSR_bit[21]);
        LFSR_bit[0] <= ~(LFSR_bit[48] ^ LFSR_bit[47] ^ LFSR_bit[21] ^ LFSR_bit[20]);
    end
    
    always @(posedge clk) begin
        if ((rst && operation_start) == 1'b0) LFSR_bit[64-1:16] <= seed;
        else LFSR_bit[64-1:16] <= LFSR_bit[48-1:0];
    end
    
    // 16-bit digital output for p-bit
    assign LFSR_out = LFSR_bit[64-1:48];
        
endmodule
