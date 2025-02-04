//////////////////////////////////////////////////////////////////////////////////
// Company: Korea University
// Engineer: Hyunjin Kim & Hyundo Jung
// 
// Create Date: 2022/09/19 11:00:27
// Design Name: Probabilistic Prime Factorization Machine
// Module Name: Candidate_sieve
// Project Name: VCBM_PA
// Target Devices: Xilinx Artix-7 (Zynq-7000)
// Tool Versions: Vivado 2020.2
// Description: Candidate sieve that selects the best candidate with 3,5,7 modulo operators.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Candidate_sieve #(
    parameter [7-1:0] max_N_digit = 7'd64)(
    input [max_N_digit/2-1:0] cand,
    output reg[max_N_digit/2-1:0] best_cand
    );
    
    reg [7-1:0] cand_plus_4_mod_3;
    reg [7-1:0] cand_plus_2_mod_3;
    reg [7-1:0] cand_mod_3;
    reg [7-1:0] cand_minus_2_mod_3;
    
    reg [8-1:0] cand_plus_4_mod_5;
    reg [8-1:0] cand_plus_2_mod_5;
    reg [8-1:0] cand_mod_5;
    reg [8-1:0] cand_minus_2_mod_5;
    
    reg [8-1:0] cand_plus_4_mod_7;
    reg [8-1:0] cand_plus_2_mod_7;
    reg [8-1:0] cand_mod_7;
    reg [8-1:0] cand_minus_2_mod_7;
    
    wire [max_N_digit/2-1:0] cand_plus_4 = cand + 4;
    wire [max_N_digit/2-1:0] cand_plus_2 = cand + 2;
    wire [max_N_digit/2-1:0] cand_minus_2 = cand - 2;
    
    wire [14-1:0] cand_plus_4_mod_3_result = cand_plus_4_mod_3 - 3 * (cand_plus_4_mod_3 / 3);
    wire [14-1:0] cand_plus_2_mod_3_result = cand_plus_2_mod_3 - 3 * (cand_plus_2_mod_3 / 3);
    wire [14-1:0] cand_mod_3_result = cand_mod_3 - 3 * (cand_mod_3 / 3);
    wire [14-1:0] cand_minus_2_mod_3_result = cand_minus_2_mod_3 - 3 * (cand_minus_2_mod_3 / 3);
    
    wire [16-1:0] cand_plus_4_mod_5_result = cand_plus_4_mod_5 - 5 * (cand_plus_4_mod_5 / 5);
    wire [16-1:0] cand_plus_2_mod_5_result = cand_plus_2_mod_5 - 5 * (cand_plus_2_mod_5 / 5);
    wire [16-1:0] cand_mod_5_result = cand_mod_5 - 5 * (cand_mod_5 / 5);
    wire [16-1:0] cand_minus_2_mod_5_result = cand_minus_2_mod_5 - 5 * (cand_minus_2_mod_5 / 5);
    
    wire [16-1:0] cand_plus_4_mod_7_result = cand_plus_4_mod_7 - 7 * (cand_plus_4_mod_7 / 7);
    wire [16-1:0] cand_plus_2_mod_7_result = cand_plus_2_mod_7 - 7 * (cand_plus_2_mod_7 / 7);
    wire [16-1:0] cand_mod_7_result = cand_mod_7 - 7 * (cand_mod_7 / 7);
    wire [16-1:0] cand_minus_2_mod_7_result = cand_minus_2_mod_7 - 7 * (cand_minus_2_mod_7 / 7);
    
    integer i;
    /*
always @(*) begin
    if ((cand_mod_3_result != 0) && (cand_mod_5_result != 0) && (cand_mod_7_result != 0)) begin
        if (cand[max_N_digit/2-1:0] == 1) best_cand <= 11;
	    else best_cand <= cand;
	end
	else if ((cand_plus_2_mod_3_result != 0) && (cand_plus_2_mod_5_result != 0) && (cand_plus_2_mod_7_result != 0)) begin
		best_cand <= cand+2;
	end
	else if ((cand_minus_2_mod_3_result != 0) && (cand_minus_2_mod_5_result != 0) && (cand_minus_2_mod_7_result != 0)) begin
		best_cand <= cand-2;
	end
	else if ((cand_plus_4_mod_3_result != 0) && (cand_plus_4_mod_5_result != 0) && (cand_plus_4_mod_7_result != 0)) begin
		best_cand <= cand+4;
	end
	else begin
		best_cand <= cand-4;
	end
	
	//////////////////// modulo 3 /////////////////////////////
    cand_plus_4_mod_3 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+2) begin 
        cand_plus_4_mod_3 = cand_plus_4_mod_3 + cand_plus_4[i];
    end
    for(i=1;i< max_N_digit/2;i=i+2) begin 
        cand_plus_4_mod_3 = cand_plus_4_mod_3 + 2* cand_plus_4[i];
    end 
    
    cand_plus_2_mod_3 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+2) begin  
        cand_plus_2_mod_3 = cand_plus_2_mod_3 + cand_plus_2[i];
    end
    for(i=1;i< max_N_digit/2;i=i+2) begin   
        cand_plus_2_mod_3 = cand_plus_2_mod_3 + 2* cand_plus_2[i];
    end 
    
    cand_mod_3 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+2) begin   
        cand_mod_3 = cand_mod_3 + cand[i];
    end
    for(i=1;i< max_N_digit/2;i=i+2) begin  
        cand_mod_3 = cand_mod_3 + 2* cand[i];
    end 
    
    cand_minus_2_mod_3 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+2) begin  
        cand_minus_2_mod_3 = cand_minus_2_mod_3 + cand_minus_2[i];
    end
    for(i=1;i< max_N_digit/2;i=i+2) begin  
        cand_minus_2_mod_3 = cand_minus_2_mod_3 + 2* cand_minus_2[i];
    end 
    
    
	//////////////////// modulo 5 /////////////////////////////
    cand_plus_4_mod_5 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+4) begin 
        cand_plus_4_mod_5 = cand_plus_4_mod_5 + cand_plus_4[i];
    end
    for(i=1;i< max_N_digit/2;i=i+4) begin 
        cand_plus_4_mod_5 = cand_plus_4_mod_5 + 2* cand_plus_4[i];
    end 
    for(i=2;i< max_N_digit/2;i=i+4) begin   
        cand_plus_4_mod_5 = cand_plus_4_mod_5 + 4*cand_plus_4[i];
    end
    for(i=3;i< max_N_digit/2;i=i+4) begin  
        cand_plus_4_mod_5 = cand_plus_4_mod_5 + 3* cand_plus_4[i];
    end 
    
    
    cand_plus_2_mod_5 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+4) begin   
        cand_plus_2_mod_5 = cand_plus_2_mod_5 + cand_plus_2[i];
    end
    for(i=1;i< max_N_digit/2;i=i+4) begin   
        cand_plus_2_mod_5 = cand_plus_2_mod_5 + 2* cand_plus_2[i];
    end 
    for(i=2;i< max_N_digit/2;i=i+4) begin  
        cand_plus_2_mod_5 = cand_plus_2_mod_5 + 4*cand_plus_2[i];
    end
    for(i=3;i< max_N_digit/2;i=i+4) begin   
        cand_plus_2_mod_5 = cand_plus_2_mod_5 + 3* cand_plus_2[i];
    end 
    
    
    cand_mod_5 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+4) begin   
        cand_mod_5 = cand_mod_5 + cand[i];
    end
    for(i=1;i< max_N_digit/2;i=i+4) begin   
        cand_mod_5 = cand_mod_5 + 2* cand[i];
    end 
    for(i=2;i< max_N_digit/2;i=i+4) begin  
        cand_mod_5 = cand_mod_5 + 4*cand[i];
    end
    for(i=3;i< max_N_digit/2;i=i+4) begin   
        cand_mod_5 = cand_mod_5 + 3* cand[i];
    end 
    
    cand_minus_2_mod_5 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+4) begin   
        cand_minus_2_mod_5 = cand_minus_2_mod_5 + cand_minus_2[i];
    end
    for(i=1;i< max_N_digit/2;i=i+4) begin   
        cand_minus_2_mod_5 = cand_minus_2_mod_5 + 2* cand_minus_2[i];
    end 
    for(i=2;i< max_N_digit/2;i=i+4) begin  
        cand_minus_2_mod_5 = cand_minus_2_mod_5 + 4*cand_minus_2[i];
    end
    for(i=3;i< max_N_digit/2;i=i+4) begin   
        cand_minus_2_mod_5 = cand_minus_2_mod_5 + 3* cand_minus_2[i];
    end 
    
    
	//////////////////// modulo 7 /////////////////////////////
    cand_plus_4_mod_7 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+3) begin 
        cand_plus_4_mod_7 = cand_plus_4_mod_7 + cand_plus_4[i];
    end
    for(i=1;i< max_N_digit/2;i=i+3) begin 
        cand_plus_4_mod_7 = cand_plus_4_mod_7 + 2* cand_plus_4[i];
    end 
    for(i=2;i< max_N_digit/2;i=i+3) begin   
        cand_plus_4_mod_7 = cand_plus_4_mod_7 + 4*cand_plus_4[i];
    end
    
    cand_plus_2_mod_7 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+3) begin 
        cand_plus_2_mod_7 = cand_plus_2_mod_7 + cand_plus_2[i];
    end
    for(i=1;i< max_N_digit/2;i=i+3) begin 
        cand_plus_2_mod_7 = cand_plus_2_mod_7 + 2* cand_plus_2[i];
    end 
    for(i=2;i< max_N_digit/2;i=i+3) begin   
        cand_plus_2_mod_7 = cand_plus_2_mod_7 + 4*cand_plus_2[i];
    end
    
    cand_mod_7 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+3) begin 
        cand_mod_7 = cand_mod_7 + cand[i];
    end
    for(i=1;i< max_N_digit/2;i=i+3) begin 
        cand_mod_7 = cand_mod_7 + 2* cand[i];
    end 
    for(i=2;i< max_N_digit/2;i=i+3) begin   
        cand_mod_7 = cand_mod_7 + 4*cand[i];
    end
    
    cand_minus_2_mod_7 = 0;  //initialize count variable.
    for(i=0;i< max_N_digit/2;i=i+3) begin 
        cand_minus_2_mod_7 = cand_minus_2_mod_7 + cand_minus_2[i];
    end
    for(i=1;i< max_N_digit/2;i=i+3) begin 
        cand_minus_2_mod_7 = cand_minus_2_mod_7 + 2* cand_minus_2[i];
    end 
    for(i=2;i< max_N_digit/2;i=i+3) begin   
        cand_minus_2_mod_7 = cand_minus_2_mod_7 + 4*cand_minus_2[i];
    end
end*/

    
    always @(*) begin
        //initialize count variable.
        cand_plus_4_mod_3 = 0;  
        cand_plus_2_mod_3 = 0;
        cand_mod_3 = 0;
        cand_minus_2_mod_3 = 0;        
        cand_plus_4_mod_5 = 0;
        cand_plus_2_mod_5 = 0;
        cand_mod_5 = 0;
        cand_minus_2_mod_5 = 0;        
        cand_plus_4_mod_7 = 0;
        cand_plus_2_mod_7 = 0;
        cand_mod_7 = 0;
        cand_minus_2_mod_7 = 0;
            
        
        /////////////////////// modulo 3 ///////////////////////
        for(i=0;i< max_N_digit/2;i=i+2) cand_plus_4_mod_3 = cand_plus_4_mod_3 + cand_plus_4[i];
        for(i=1;i< max_N_digit/2;i=i+2) cand_plus_4_mod_3 = cand_plus_4_mod_3 + 2* cand_plus_4[i];
        for(i=0;i< max_N_digit/2;i=i+2) cand_plus_2_mod_3 = cand_plus_2_mod_3 + cand_plus_2[i];
        for(i=1;i< max_N_digit/2;i=i+2) cand_plus_2_mod_3 = cand_plus_2_mod_3 + 2* cand_plus_2[i];
        for(i=0;i< max_N_digit/2;i=i+2) cand_mod_3 = cand_mod_3 + cand[i];
        for(i=1;i< max_N_digit/2;i=i+2) cand_mod_3 = cand_mod_3 + 2* cand[i];
        for(i=0;i< max_N_digit/2;i=i+2) cand_minus_2_mod_3 = cand_minus_2_mod_3 + cand_minus_2[i];
        for(i=1;i< max_N_digit/2;i=i+2) cand_minus_2_mod_3 = cand_minus_2_mod_3 + 2* cand_minus_2[i];
        
        /////////////////////// modulo 5 ///////////////////////
        for(i=0;i< max_N_digit/2;i=i+4) cand_plus_4_mod_5 = cand_plus_4_mod_5 + cand_plus_4[i];
        for(i=1;i< max_N_digit/2;i=i+4) cand_plus_4_mod_5 = cand_plus_4_mod_5 + 2* cand_plus_4[i];
        for(i=2;i< max_N_digit/2;i=i+4) cand_plus_4_mod_5 = cand_plus_4_mod_5 + 4*cand_plus_4[i];
        for(i=3;i< max_N_digit/2;i=i+4) cand_plus_4_mod_5 = cand_plus_4_mod_5 + 3* cand_plus_4[i];
        for(i=0;i< max_N_digit/2;i=i+4) cand_plus_2_mod_5 = cand_plus_2_mod_5 + cand_plus_2[i];
        for(i=1;i< max_N_digit/2;i=i+4) cand_plus_2_mod_5 = cand_plus_2_mod_5 + 2* cand_plus_2[i];
        for(i=2;i< max_N_digit/2;i=i+4) cand_plus_2_mod_5 = cand_plus_2_mod_5 + 4*cand_plus_2[i];
        for(i=3;i< max_N_digit/2;i=i+4) cand_plus_2_mod_5 = cand_plus_2_mod_5 + 3* cand_plus_2[i];
        for(i=0;i< max_N_digit/2;i=i+4) cand_mod_5 = cand_mod_5 + cand[i];
        for(i=1;i< max_N_digit/2;i=i+4) cand_mod_5 = cand_mod_5 + 2* cand[i];
        for(i=2;i< max_N_digit/2;i=i+4) cand_mod_5 = cand_mod_5 + 4*cand[i];
        for(i=3;i< max_N_digit/2;i=i+4) cand_mod_5 = cand_mod_5 + 3* cand[i];
        for(i=0;i< max_N_digit/2;i=i+4) cand_minus_2_mod_5 = cand_minus_2_mod_5 + cand_minus_2[i];
        for(i=1;i< max_N_digit/2;i=i+4) cand_minus_2_mod_5 = cand_minus_2_mod_5 + 2* cand_minus_2[i];
        for(i=2;i< max_N_digit/2;i=i+4) cand_minus_2_mod_5 = cand_minus_2_mod_5 + 4*cand_minus_2[i];
        for(i=3;i< max_N_digit/2;i=i+4) cand_minus_2_mod_5 = cand_minus_2_mod_5 + 3* cand_minus_2[i];
        
        /////////////////////// modulo 7 ///////////////////////
        for(i=0;i< max_N_digit/2;i=i+3) cand_plus_4_mod_7 = cand_plus_4_mod_7 + cand_plus_4[i];
        for(i=1;i< max_N_digit/2;i=i+3) cand_plus_4_mod_7 = cand_plus_4_mod_7 + 2* cand_plus_4[i];
        for(i=2;i< max_N_digit/2;i=i+3) cand_plus_4_mod_7 = cand_plus_4_mod_7 + 4*cand_plus_4[i];
        for(i=0;i< max_N_digit/2;i=i+3) cand_plus_2_mod_7 = cand_plus_2_mod_7 + cand_plus_2[i];
        for(i=1;i< max_N_digit/2;i=i+3) cand_plus_2_mod_7 = cand_plus_2_mod_7 + 2* cand_plus_2[i];
        for(i=2;i< max_N_digit/2;i=i+3) cand_plus_2_mod_7 = cand_plus_2_mod_7 + 4*cand_plus_2[i];
        for(i=0;i< max_N_digit/2;i=i+3) cand_mod_7 = cand_mod_7 + cand[i];
        for(i=1;i< max_N_digit/2;i=i+3) cand_mod_7 = cand_mod_7 + 2* cand[i];
        for(i=2;i< max_N_digit/2;i=i+3) cand_mod_7 = cand_mod_7 + 4*cand[i];
        for(i=0;i< max_N_digit/2;i=i+3) cand_minus_2_mod_7 = cand_minus_2_mod_7 + cand_minus_2[i];
        for(i=1;i< max_N_digit/2;i=i+3) cand_minus_2_mod_7 = cand_minus_2_mod_7 + 2* cand_minus_2[i];
        for(i=2;i< max_N_digit/2;i=i+3) cand_minus_2_mod_7 = cand_minus_2_mod_7 + 4*cand_minus_2[i];
        
        if ((cand_mod_3_result != 0) && (cand_mod_5_result != 0) && (cand_mod_7_result != 0)) begin
            if (cand[max_N_digit/2-1:0] == 1) best_cand <= 11;
            else best_cand <= cand;
        end
        else if ((cand_plus_2_mod_3_result != 0) && (cand_plus_2_mod_5_result != 0) && (cand_plus_2_mod_7_result != 0)) best_cand <= cand+2;
        else if ((cand_minus_2_mod_3_result != 0) && (cand_minus_2_mod_5_result != 0) && (cand_minus_2_mod_7_result != 0)) best_cand <= cand-2;
        else if ((cand_plus_4_mod_3_result != 0) && (cand_plus_4_mod_5_result != 0) && (cand_plus_4_mod_7_result != 0)) best_cand <= cand+4;
        else begin
            if (cand[max_N_digit/2-1:0] == 3) best_cand <= 11;
            else best_cand <= cand-4;
        end
    end
        
endmodule
