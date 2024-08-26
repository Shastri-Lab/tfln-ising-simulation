/******************************************************************************
* Company: Korea University
* Company: Korea University
* Engineer: Hyunjin Kim & Hyundo Jung
* 
* Create Date: 2022/09/19 11:00:27
* Design Name: Probabilistic Prime Factorization Machine
* Module Name: Vitis_Model_with_decision
* Project Name: VCBM_PA
* Tool Versions: Xilinx Vitis 2020.2
* Description: Vitis code to give input of Model_with_decision_and_sieve
* 
* Dependencies: 
* 
* Revision:
* Revision 0.01 - File Created
* Additional Comments:
* 
******************************************************************************/

/*
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xil_io.h"

//operating mode
#define WRITE 1
#define READ 2

#define AXI_DATA_BYTE 4

int main() {
    int data;
    int reg_num;
    while (1) {
    	printf("======= Set Simulation ======\n");
    	printf("please input run mode\n");
    	printf("1. write\n");
    	printf("2. read\n");
    	scanf("%d",&data);
    	fflush(stdin);
    	//write operate
    	if(data == WRITE){
    		printf("please input WRITE register number (0~3)\n");
    		scanf("%d",&reg_num);
    		xil_printf("%d please input Value\r\n", reg_num);
    		scanf("%d",&data);
    		Xil_Out32((XPAR_AXI_DECISION_AND_SIEVE_0_BASEADDR) + (reg_num*AXI_DATA_BYTE), (u32)(data));
    		printf("register write done register_number (%d), value : %d\r\n", reg_num, data);

    	}
    	//read operate
    	else if (data == READ){
    		printf("please input READ register number (0~3)\n");
    		scanf("%d",&reg_num);
    		data = Xil_In32((XPAR_AXI_DECISION_AND_SIEVE_0_BASEADDR) + (reg_num*AXI_DATA_BYTE));
    		xil_printf("register read done register_number (%d), value : %d\r\n", reg_num, data);
    	} else {
    		// no operation, exit
    		//break;
    	}
    	data = 0;
    }
    return 0;
}
