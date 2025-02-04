set_property -dict {PACKAGE_PIN G15 IOSTANDARD LVCMOS33} [get_ports i_operation_start_0]

create_debug_core u_ila_0 ila
set_property ALL_PROBE_SAME_MU true [get_debug_cores u_ila_0]
set_property ALL_PROBE_SAME_MU_CNT 1 [get_debug_cores u_ila_0]
set_property C_ADV_TRIGGER false [get_debug_cores u_ila_0]
set_property C_DATA_DEPTH 1024 [get_debug_cores u_ila_0]
set_property C_EN_STRG_QUAL false [get_debug_cores u_ila_0]
set_property C_INPUT_PIPE_STAGES 3 [get_debug_cores u_ila_0]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]
set_property port_width 1 [get_debug_ports u_ila_0/clk]
connect_debug_port u_ila_0/clk [get_nets [list design_1_i/clk_wiz_0/inst/clk_out2]]
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe0]
set_property port_width 31 [get_debug_ports u_ila_0/probe0]
connect_debug_port u_ila_0/probe0 [get_nets [list {design_1_i/Top_final_0_X[1]} {design_1_i/Top_final_0_X[2]} {design_1_i/Top_final_0_X[3]} {design_1_i/Top_final_0_X[4]} {design_1_i/Top_final_0_X[5]} {design_1_i/Top_final_0_X[6]} {design_1_i/Top_final_0_X[7]} {design_1_i/Top_final_0_X[8]} {design_1_i/Top_final_0_X[9]} {design_1_i/Top_final_0_X[10]} {design_1_i/Top_final_0_X[11]} {design_1_i/Top_final_0_X[12]} {design_1_i/Top_final_0_X[13]} {design_1_i/Top_final_0_X[14]} {design_1_i/Top_final_0_X[15]} {design_1_i/Top_final_0_X[16]} {design_1_i/Top_final_0_X[17]} {design_1_i/Top_final_0_X[18]} {design_1_i/Top_final_0_X[19]} {design_1_i/Top_final_0_X[20]} {design_1_i/Top_final_0_X[21]} {design_1_i/Top_final_0_X[22]} {design_1_i/Top_final_0_X[23]} {design_1_i/Top_final_0_X[24]} {design_1_i/Top_final_0_X[25]} {design_1_i/Top_final_0_X[26]} {design_1_i/Top_final_0_X[27]} {design_1_i/Top_final_0_X[28]} {design_1_i/Top_final_0_X[29]} {design_1_i/Top_final_0_X[30]} {design_1_i/Top_final_0_X[31]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe1]
set_property port_width 31 [get_debug_ports u_ila_0/probe1]
connect_debug_port u_ila_0/probe1 [get_nets [list {design_1_i/Top_final_0_Y[1]} {design_1_i/Top_final_0_Y[2]} {design_1_i/Top_final_0_Y[3]} {design_1_i/Top_final_0_Y[4]} {design_1_i/Top_final_0_Y[5]} {design_1_i/Top_final_0_Y[6]} {design_1_i/Top_final_0_Y[7]} {design_1_i/Top_final_0_Y[8]} {design_1_i/Top_final_0_Y[9]} {design_1_i/Top_final_0_Y[10]} {design_1_i/Top_final_0_Y[11]} {design_1_i/Top_final_0_Y[12]} {design_1_i/Top_final_0_Y[13]} {design_1_i/Top_final_0_Y[14]} {design_1_i/Top_final_0_Y[15]} {design_1_i/Top_final_0_Y[16]} {design_1_i/Top_final_0_Y[17]} {design_1_i/Top_final_0_Y[18]} {design_1_i/Top_final_0_Y[19]} {design_1_i/Top_final_0_Y[20]} {design_1_i/Top_final_0_Y[21]} {design_1_i/Top_final_0_Y[22]} {design_1_i/Top_final_0_Y[23]} {design_1_i/Top_final_0_Y[24]} {design_1_i/Top_final_0_Y[25]} {design_1_i/Top_final_0_Y[26]} {design_1_i/Top_final_0_Y[27]} {design_1_i/Top_final_0_Y[28]} {design_1_i/Top_final_0_Y[29]} {design_1_i/Top_final_0_Y[30]} {design_1_i/Top_final_0_Y[31]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe2]
set_property port_width 32 [get_debug_ports u_ila_0/probe2]
connect_debug_port u_ila_0/probe2 [get_nets [list {design_1_i/Top_final_0_operation_count[0]} {design_1_i/Top_final_0_operation_count[1]} {design_1_i/Top_final_0_operation_count[2]} {design_1_i/Top_final_0_operation_count[3]} {design_1_i/Top_final_0_operation_count[4]} {design_1_i/Top_final_0_operation_count[5]} {design_1_i/Top_final_0_operation_count[6]} {design_1_i/Top_final_0_operation_count[7]} {design_1_i/Top_final_0_operation_count[8]} {design_1_i/Top_final_0_operation_count[9]} {design_1_i/Top_final_0_operation_count[10]} {design_1_i/Top_final_0_operation_count[11]} {design_1_i/Top_final_0_operation_count[12]} {design_1_i/Top_final_0_operation_count[13]} {design_1_i/Top_final_0_operation_count[14]} {design_1_i/Top_final_0_operation_count[15]} {design_1_i/Top_final_0_operation_count[16]} {design_1_i/Top_final_0_operation_count[17]} {design_1_i/Top_final_0_operation_count[18]} {design_1_i/Top_final_0_operation_count[19]} {design_1_i/Top_final_0_operation_count[20]} {design_1_i/Top_final_0_operation_count[21]} {design_1_i/Top_final_0_operation_count[22]} {design_1_i/Top_final_0_operation_count[23]} {design_1_i/Top_final_0_operation_count[24]} {design_1_i/Top_final_0_operation_count[25]} {design_1_i/Top_final_0_operation_count[26]} {design_1_i/Top_final_0_operation_count[27]} {design_1_i/Top_final_0_operation_count[28]} {design_1_i/Top_final_0_operation_count[29]} {design_1_i/Top_final_0_operation_count[30]} {design_1_i/Top_final_0_operation_count[31]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe3]
set_property port_width 30 [get_debug_ports u_ila_0/probe3]
connect_debug_port u_ila_0/probe3 [get_nets [list {design_1_i/axi_final64_0_o_N_LSB[0]} {design_1_i/axi_final64_0_o_N_LSB[1]} {design_1_i/axi_final64_0_o_N_LSB[2]} {design_1_i/axi_final64_0_o_N_LSB[3]} {design_1_i/axi_final64_0_o_N_LSB[4]} {design_1_i/axi_final64_0_o_N_LSB[5]} {design_1_i/axi_final64_0_o_N_LSB[6]} {design_1_i/axi_final64_0_o_N_LSB[7]} {design_1_i/axi_final64_0_o_N_LSB[8]} {design_1_i/axi_final64_0_o_N_LSB[9]} {design_1_i/axi_final64_0_o_N_LSB[10]} {design_1_i/axi_final64_0_o_N_LSB[11]} {design_1_i/axi_final64_0_o_N_LSB[12]} {design_1_i/axi_final64_0_o_N_LSB[13]} {design_1_i/axi_final64_0_o_N_LSB[14]} {design_1_i/axi_final64_0_o_N_LSB[15]} {design_1_i/axi_final64_0_o_N_LSB[16]} {design_1_i/axi_final64_0_o_N_LSB[17]} {design_1_i/axi_final64_0_o_N_LSB[18]} {design_1_i/axi_final64_0_o_N_LSB[19]} {design_1_i/axi_final64_0_o_N_LSB[20]} {design_1_i/axi_final64_0_o_N_LSB[21]} {design_1_i/axi_final64_0_o_N_LSB[22]} {design_1_i/axi_final64_0_o_N_LSB[23]} {design_1_i/axi_final64_0_o_N_LSB[24]} {design_1_i/axi_final64_0_o_N_LSB[25]} {design_1_i/axi_final64_0_o_N_LSB[26]} {design_1_i/axi_final64_0_o_N_LSB[27]} {design_1_i/axi_final64_0_o_N_LSB[28]} {design_1_i/axi_final64_0_o_N_LSB[29]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe4]
set_property port_width 30 [get_debug_ports u_ila_0/probe4]
connect_debug_port u_ila_0/probe4 [get_nets [list {design_1_i/axi_final64_0_o_N_MID[0]} {design_1_i/axi_final64_0_o_N_MID[1]} {design_1_i/axi_final64_0_o_N_MID[2]} {design_1_i/axi_final64_0_o_N_MID[3]} {design_1_i/axi_final64_0_o_N_MID[4]} {design_1_i/axi_final64_0_o_N_MID[5]} {design_1_i/axi_final64_0_o_N_MID[6]} {design_1_i/axi_final64_0_o_N_MID[7]} {design_1_i/axi_final64_0_o_N_MID[8]} {design_1_i/axi_final64_0_o_N_MID[9]} {design_1_i/axi_final64_0_o_N_MID[10]} {design_1_i/axi_final64_0_o_N_MID[11]} {design_1_i/axi_final64_0_o_N_MID[12]} {design_1_i/axi_final64_0_o_N_MID[13]} {design_1_i/axi_final64_0_o_N_MID[14]} {design_1_i/axi_final64_0_o_N_MID[15]} {design_1_i/axi_final64_0_o_N_MID[16]} {design_1_i/axi_final64_0_o_N_MID[17]} {design_1_i/axi_final64_0_o_N_MID[18]} {design_1_i/axi_final64_0_o_N_MID[19]} {design_1_i/axi_final64_0_o_N_MID[20]} {design_1_i/axi_final64_0_o_N_MID[21]} {design_1_i/axi_final64_0_o_N_MID[22]} {design_1_i/axi_final64_0_o_N_MID[23]} {design_1_i/axi_final64_0_o_N_MID[24]} {design_1_i/axi_final64_0_o_N_MID[25]} {design_1_i/axi_final64_0_o_N_MID[26]} {design_1_i/axi_final64_0_o_N_MID[27]} {design_1_i/axi_final64_0_o_N_MID[28]} {design_1_i/axi_final64_0_o_N_MID[29]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe5]
set_property port_width 4 [get_debug_ports u_ila_0/probe5]
connect_debug_port u_ila_0/probe5 [get_nets [list {design_1_i/axi_final64_0_o_N_MSB[0]} {design_1_i/axi_final64_0_o_N_MSB[1]} {design_1_i/axi_final64_0_o_N_MSB[2]} {design_1_i/axi_final64_0_o_N_MSB[3]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe6]
set_property port_width 32 [get_debug_ports u_ila_0/probe6]
connect_debug_port u_ila_0/probe6 [get_nets [list {design_1_i/axi_final64_0_o_seed[0]} {design_1_i/axi_final64_0_o_seed[1]} {design_1_i/axi_final64_0_o_seed[2]} {design_1_i/axi_final64_0_o_seed[3]} {design_1_i/axi_final64_0_o_seed[4]} {design_1_i/axi_final64_0_o_seed[5]} {design_1_i/axi_final64_0_o_seed[6]} {design_1_i/axi_final64_0_o_seed[7]} {design_1_i/axi_final64_0_o_seed[8]} {design_1_i/axi_final64_0_o_seed[9]} {design_1_i/axi_final64_0_o_seed[10]} {design_1_i/axi_final64_0_o_seed[11]} {design_1_i/axi_final64_0_o_seed[12]} {design_1_i/axi_final64_0_o_seed[13]} {design_1_i/axi_final64_0_o_seed[14]} {design_1_i/axi_final64_0_o_seed[15]} {design_1_i/axi_final64_0_o_seed[16]} {design_1_i/axi_final64_0_o_seed[17]} {design_1_i/axi_final64_0_o_seed[18]} {design_1_i/axi_final64_0_o_seed[19]} {design_1_i/axi_final64_0_o_seed[20]} {design_1_i/axi_final64_0_o_seed[21]} {design_1_i/axi_final64_0_o_seed[22]} {design_1_i/axi_final64_0_o_seed[23]} {design_1_i/axi_final64_0_o_seed[24]} {design_1_i/axi_final64_0_o_seed[25]} {design_1_i/axi_final64_0_o_seed[26]} {design_1_i/axi_final64_0_o_seed[27]} {design_1_i/axi_final64_0_o_seed[28]} {design_1_i/axi_final64_0_o_seed[29]} {design_1_i/axi_final64_0_o_seed[30]} {design_1_i/axi_final64_0_o_seed[31]}]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe7]
set_property port_width 1 [get_debug_ports u_ila_0/probe7]
connect_debug_port u_ila_0/probe7 [get_nets [list design_1_i/i_operation_start_0_1]]
create_debug_port u_ila_0 probe
set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe8]
set_property port_width 1 [get_debug_ports u_ila_0/probe8]
connect_debug_port u_ila_0/probe8 [get_nets [list design_1_i/Top_final_0_operation_end]]
set_property C_CLK_INPUT_FREQ_HZ 300000000 [get_debug_cores dbg_hub]
set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]
connect_debug_port dbg_hub/clk [get_nets clk]
