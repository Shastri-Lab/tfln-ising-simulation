//////////////////////////////////////////////////////////////////////////////////
// Company: Korea University
// Engineer: Hyunjin Kim & Hyundo Jung
// 
// Create Date: 2022/09/19 11:00:27
// Design Name: Probabilistic Prime Factorization Machine
// Module Name: Sigmoid_LUT
// Project Name: VCBM_PA
// Target Devices: Xilinx Artix-7 (Zynq-7000)
// Tool Versions: Vivado 2020.2
// Description: Lookup table-based 8-bit input & 16-bit output sigmoid function for p-bit.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Sigmoid_LUT(
    input [8-1:0] Sigmoid_in,
    output reg [16-1:0] Sigmoid_out
    );
    
    reg [16-1:0] Sigmoid_out_reg;
    
    always @(*) begin
        // changes sigmoid output considering the signed bit
        if (Sigmoid_in[7]) Sigmoid_out <= Sigmoid_out_reg;
        else Sigmoid_out <= 16'b1111_1111_1111_1111 - Sigmoid_out_reg;
        
        // -8 ~ +8 8-bit input & 0 ~ 1 15-bit output sigmoid function
        case (Sigmoid_in[7-1:0])
            7'b000_0000: Sigmoid_out_reg <= 16'h0001;
            7'b000_0001: Sigmoid_out_reg <= 16'h0017;
            7'b000_0010: Sigmoid_out_reg <= 16'h0019;
            7'b000_0011: Sigmoid_out_reg <= 16'h001b;
            7'b000_0100: Sigmoid_out_reg <= 16'h001c;
            7'b000_0101: Sigmoid_out_reg <= 16'h001e;
            7'b000_0110: Sigmoid_out_reg <= 16'h0020;
            7'b000_0111: Sigmoid_out_reg <= 16'h0022;
            7'b000_1000: Sigmoid_out_reg <= 16'h0024;
            7'b000_1001: Sigmoid_out_reg <= 16'h0027;
            7'b000_1010: Sigmoid_out_reg <= 16'h0029;
            7'b000_1011: Sigmoid_out_reg <= 16'h002c;
            7'b000_1100: Sigmoid_out_reg <= 16'h002f;
            7'b000_1101: Sigmoid_out_reg <= 16'h0032;
            7'b000_1110: Sigmoid_out_reg <= 16'h0035;
            7'b000_1111: Sigmoid_out_reg <= 16'h0038;
            7'b001_0000: Sigmoid_out_reg <= 16'h003c;
            7'b001_0001: Sigmoid_out_reg <= 16'h0040;
            7'b001_0010: Sigmoid_out_reg <= 16'h0044;
            7'b001_0011: Sigmoid_out_reg <= 16'h0048;
            7'b001_0100: Sigmoid_out_reg <= 16'h004d;
            7'b001_0101: Sigmoid_out_reg <= 16'h0052;
            7'b001_0110: Sigmoid_out_reg <= 16'h0057;
            7'b001_0111: Sigmoid_out_reg <= 16'h005c;
            7'b001_1000: Sigmoid_out_reg <= 16'h0062;
            7'b001_1001: Sigmoid_out_reg <= 16'h0069;
            7'b001_1010: Sigmoid_out_reg <= 16'h006f;
            7'b001_1011: Sigmoid_out_reg <= 16'h0077;
            7'b001_1100: Sigmoid_out_reg <= 16'h007e;
            7'b001_1101: Sigmoid_out_reg <= 16'h0086;
            7'b001_1110: Sigmoid_out_reg <= 16'h008f;
            7'b001_1111: Sigmoid_out_reg <= 16'h0098;
            7'b010_0000: Sigmoid_out_reg <= 16'h00a2;
            7'b010_0001: Sigmoid_out_reg <= 16'h00ac;
            7'b010_0010: Sigmoid_out_reg <= 16'h00b8;
            7'b010_0011: Sigmoid_out_reg <= 16'h00c3;
            7'b010_0100: Sigmoid_out_reg <= 16'h00d0;
            7'b010_0101: Sigmoid_out_reg <= 16'h00dd;
            7'b010_0110: Sigmoid_out_reg <= 16'h00ec;
            7'b010_0111: Sigmoid_out_reg <= 16'h00fb;
            7'b010_1000: Sigmoid_out_reg <= 16'h010b;
            7'b010_1001: Sigmoid_out_reg <= 16'h011c;
            7'b010_1010: Sigmoid_out_reg <= 16'h012e;
            7'b010_1011: Sigmoid_out_reg <= 16'h0141;
            7'b010_1100: Sigmoid_out_reg <= 16'h0156;
            7'b010_1101: Sigmoid_out_reg <= 16'h016c;
            7'b010_1110: Sigmoid_out_reg <= 16'h0183;
            7'b010_1111: Sigmoid_out_reg <= 16'h019c;
            7'b011_0000: Sigmoid_out_reg <= 16'h01b7;
            7'b011_0001: Sigmoid_out_reg <= 16'h01d3;
            7'b011_0010: Sigmoid_out_reg <= 16'h01f1;
            7'b011_0011: Sigmoid_out_reg <= 16'h0210;
            7'b011_0100: Sigmoid_out_reg <= 16'h0232;
            7'b011_0101: Sigmoid_out_reg <= 16'h0256;
            7'b011_0110: Sigmoid_out_reg <= 16'h027c;
            7'b011_0111: Sigmoid_out_reg <= 16'h02a5;
            7'b011_1000: Sigmoid_out_reg <= 16'h02d0;
            7'b011_1001: Sigmoid_out_reg <= 16'h02fe;
            7'b011_1010: Sigmoid_out_reg <= 16'h032f;
            7'b011_1011: Sigmoid_out_reg <= 16'h0363;
            7'b011_1100: Sigmoid_out_reg <= 16'h039a;
            7'b011_1101: Sigmoid_out_reg <= 16'h03d4;
            7'b011_1110: Sigmoid_out_reg <= 16'h0412;
            7'b011_1111: Sigmoid_out_reg <= 16'h0455;
            7'b100_0000: Sigmoid_out_reg <= 16'h049b;
            7'b100_0001: Sigmoid_out_reg <= 16'h04e5;
            7'b100_0010: Sigmoid_out_reg <= 16'h0535;
            7'b100_0011: Sigmoid_out_reg <= 16'h0589;
            7'b100_0100: Sigmoid_out_reg <= 16'h05e2;
            7'b100_0101: Sigmoid_out_reg <= 16'h0641;
            7'b100_0110: Sigmoid_out_reg <= 16'h06a5;
            7'b100_0111: Sigmoid_out_reg <= 16'h0710;
            7'b100_1000: Sigmoid_out_reg <= 16'h0781;
            7'b100_1001: Sigmoid_out_reg <= 16'h07f9;
            7'b100_1010: Sigmoid_out_reg <= 16'h0878;
            7'b100_1011: Sigmoid_out_reg <= 16'h08ff;
            7'b100_1100: Sigmoid_out_reg <= 16'h098e;
            7'b100_1101: Sigmoid_out_reg <= 16'h0a26;
            7'b100_1110: Sigmoid_out_reg <= 16'h0ac6;
            7'b100_1111: Sigmoid_out_reg <= 16'h0b70;
            7'b101_0000: Sigmoid_out_reg <= 16'h0c24;
            7'b101_0001: Sigmoid_out_reg <= 16'h0ce2;
            7'b101_0010: Sigmoid_out_reg <= 16'h0dac;
            7'b101_0011: Sigmoid_out_reg <= 16'h0e81;
            7'b101_0100: Sigmoid_out_reg <= 16'h0f62;
            7'b101_0101: Sigmoid_out_reg <= 16'h1050;
            7'b101_0110: Sigmoid_out_reg <= 16'h114b;
            7'b101_0111: Sigmoid_out_reg <= 16'h1254;
            7'b101_1000: Sigmoid_out_reg <= 16'h136b;
            7'b101_1001: Sigmoid_out_reg <= 16'h1492;
            7'b101_1010: Sigmoid_out_reg <= 16'h15c9;
            7'b101_1011: Sigmoid_out_reg <= 16'h1710;
            7'b101_1100: Sigmoid_out_reg <= 16'h1869;
            7'b101_1101: Sigmoid_out_reg <= 16'h19d3;
            7'b101_1110: Sigmoid_out_reg <= 16'h1b50;
            7'b101_1111: Sigmoid_out_reg <= 16'h1ce0;
            7'b110_0000: Sigmoid_out_reg <= 16'h1e84;
            7'b110_0001: Sigmoid_out_reg <= 16'h203c;
            7'b110_0010: Sigmoid_out_reg <= 16'h220a;
            7'b110_0011: Sigmoid_out_reg <= 16'h23ed;
            7'b110_0100: Sigmoid_out_reg <= 16'h25e6;
            7'b110_0101: Sigmoid_out_reg <= 16'h27f6;
            7'b110_0110: Sigmoid_out_reg <= 16'h2a1e;
            7'b110_0111: Sigmoid_out_reg <= 16'h2c5d;
            7'b110_1000: Sigmoid_out_reg <= 16'h2eb3;
            7'b110_1001: Sigmoid_out_reg <= 16'h3123;
            7'b110_1010: Sigmoid_out_reg <= 16'h33aa;
            7'b110_1011: Sigmoid_out_reg <= 16'h364a;
            7'b110_1100: Sigmoid_out_reg <= 16'h3903;
            7'b110_1101: Sigmoid_out_reg <= 16'h3bd4;
            7'b110_1110: Sigmoid_out_reg <= 16'h3ebe;
            7'b110_1111: Sigmoid_out_reg <= 16'h41c0;
            7'b111_0000: Sigmoid_out_reg <= 16'h44d9;
            7'b111_0001: Sigmoid_out_reg <= 16'h480a;
            7'b111_0010: Sigmoid_out_reg <= 16'h4b52;
            7'b111_0011: Sigmoid_out_reg <= 16'h4eaf;
            7'b111_0100: Sigmoid_out_reg <= 16'h5221;
            7'b111_0101: Sigmoid_out_reg <= 16'h55a8;
            7'b111_0110: Sigmoid_out_reg <= 16'h5941;
            7'b111_0111: Sigmoid_out_reg <= 16'h5cec;
            7'b111_1000: Sigmoid_out_reg <= 16'h60a7;
            7'b111_1001: Sigmoid_out_reg <= 16'h6470;
            7'b111_1010: Sigmoid_out_reg <= 16'h6847;
            7'b111_1011: Sigmoid_out_reg <= 16'h6c29;
            7'b111_1100: Sigmoid_out_reg <= 16'h7015;
            7'b111_1101: Sigmoid_out_reg <= 16'h7409;
            7'b111_1110: Sigmoid_out_reg <= 16'h7803;
            7'b111_1111: Sigmoid_out_reg <= 16'h7c00;
        endcase
    end
    
endmodule
