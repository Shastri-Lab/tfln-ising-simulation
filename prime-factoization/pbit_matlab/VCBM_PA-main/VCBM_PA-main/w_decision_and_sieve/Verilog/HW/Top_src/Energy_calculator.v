//////////////////////////////////////////////////////////////////////////////////
// Company: Korea University
// Engineer: Hyunjin Kim & Hyundo Jung
// 
// Create Date: 2022/09/19 11:00:27
// Design Name: Probabilistic Prime Factorization Machine
// Module Name: Energy_calculator
// Project Name: VCBM_PA
// Target Devices: Xilinx Artix-7 (Zynq-7000)
// Tool Versions: Vivado 2020.2
// Description: Energy_calculator for updating X in virtually connected Boltzmann Machine.
//             The calculator generates probability input of each p-bit.
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Energy_calculator #(
    parameter [7-1:0] max_N_digit = 7'd64) (
    input [max_N_digit/2-1:0] X,
    input [max_N_digit/2-1:0] Y,
    input [max_N_digit-1:0] N,
    input [2-1:0] energy_shift,
    input [7-1:0] N_digit,
    output reg [max_N_digit/2-1-1:0] pbit_in_0,
    output reg [max_N_digit/2-1-1:0] pbit_in_1,
    output reg [max_N_digit/2-1-1:0] pbit_in_2,
    output reg [max_N_digit/2-1-1:0] pbit_in_3,
    output reg [max_N_digit/2-1-1:0] pbit_in_4,
    output reg [max_N_digit/2-1-1:0] pbit_in_5,
    output reg [max_N_digit/2-1-1:0] pbit_in_6,
    output reg [max_N_digit/2-1-1:0] pbit_in_7
    );
    
    wire signed [max_N_digit/2+1-1:0] Y_s = {1'b0, Y};
    wire signed [max_N_digit+1-1:0] N_s = {1'b0, N};
    wire [max_N_digit-1:0] XY = X * Y;
    wire signed [max_N_digit+1-1:0] XY_s = {1'b0, XY};
    
    wire signed [max_N_digit+1-1:0] energy_cal_1 = (N_s - XY_s);
    wire signed [3*max_N_digit/2+2-1:0] energy_cal_2 = (energy_cal_1*Y_s);
    wire signed [3*max_N_digit/2-5-1:0] energy_cal_3 = energy_cal_2[3*max_N_digit/2+2-1:7];
    
    wire [max_N_digit - 1:0] Y_square_before = (Y * Y);
    wire [max_N_digit - 6 - 1:0] Y_square = Y_square_before[max_N_digit - 1:6];
    wire signed [max_N_digit - 5 -1:0] Y_square_s = {1'b0, Y_square};  
    
    genvar k; // k means digital bit of X
    
    for (k = 2; k < max_N_digit/2+1; k = k +1) begin
    
        // 1 signed bit, 3 integer bit, and 4 fractional bit
        wire signed [3*max_N_digit/2-5-1:0] energy_cal_system = energy_cal_3 >>> (2*N_digit - 3 - energy_shift - k - 7 - 4);
        wire signed [max_N_digit-5-1:0] Y_square_k = Y_square_s >>> (2*N_digit - 1 - energy_shift - 2*k - 6 - 4);
        wire signed [3*max_N_digit/2-4-1:0] energy_cal = (X[k-1]) ? energy_cal_system + Y_square_k : energy_cal_system - Y_square_k;
        wire [7-1:0] energy_cal_unsigned = ~energy_cal[7-1:0] + 1;
        
        always @(energy_cal) begin
            if (energy_cal[3*max_N_digit/2-4-1]) begin
                if ((-energy_cal[3*max_N_digit/2-4-1:0])>127) begin
                    pbit_in_0[k-2] <= 0;
                    pbit_in_1[k-2] <= 0;
                    pbit_in_2[k-2] <= 0;
                    pbit_in_3[k-2] <= 0;
                    pbit_in_4[k-2] <= 0;
                    pbit_in_5[k-2] <= 0;
                    pbit_in_6[k-2] <= 0;
                    pbit_in_7[k-2] <= 1;
                end
                else begin
                    pbit_in_0[k-2] <= energy_cal[0];
                    pbit_in_1[k-2] <= energy_cal[1];
                    pbit_in_2[k-2] <= energy_cal[2];
                    pbit_in_3[k-2] <= energy_cal[3];
                    pbit_in_4[k-2] <= energy_cal[4];
                    pbit_in_5[k-2] <= energy_cal[5];
                    pbit_in_6[k-2] <= energy_cal[6];
                    pbit_in_7[k-2] <= 1;
                end
            end
            
            else begin
                if (energy_cal[3*max_N_digit/2-4-1:0]>126) begin
                    pbit_in_0[k-2] <= 0;
                    pbit_in_1[k-2] <= 0;
                    pbit_in_2[k-2] <= 0;
                    pbit_in_3[k-2] <= 0;
                    pbit_in_4[k-2] <= 0;
                    pbit_in_5[k-2] <= 0;
                    pbit_in_6[k-2] <= 0;
                    pbit_in_7[k-2] <= 0;
                end
                else if (energy_cal[3*max_N_digit/2-4-1:0]==0) begin
                    pbit_in_0[k-2] <= 1;
                    pbit_in_1[k-2] <= 1;
                    pbit_in_2[k-2] <= 1;
                    pbit_in_3[k-2] <= 1;
                    pbit_in_4[k-2] <= 1;
                    pbit_in_5[k-2] <= 1;
                    pbit_in_6[k-2] <= 1;
                    pbit_in_7[k-2] <= 0;
                end
                else begin
                    pbit_in_0[k-2] <= energy_cal_unsigned[0];
                    pbit_in_1[k-2] <= energy_cal_unsigned[1];
                    pbit_in_2[k-2] <= energy_cal_unsigned[2];
                    pbit_in_3[k-2] <= energy_cal_unsigned[3];
                    pbit_in_4[k-2] <= energy_cal_unsigned[4];
                    pbit_in_5[k-2] <= energy_cal_unsigned[5];
                    pbit_in_6[k-2] <= energy_cal_unsigned[6];
                    pbit_in_7[k-2] <= 0;
                end
            end
        end
    end
    
endmodule
