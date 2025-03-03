import numpy as np

# define function to compute power consumption
def compute_power(
    I_th=15e-3,  # laser threshold current (A)
    gamma_L=0.29,  # laser slope efficiency (W/A)
    V_L=1.6,  # laser bias voltage (V)
    P_o=47.28e-3,  # optical power output (W)
    
    V_PP=2.7,  # modulator drive voltage (Vpp)
    R=50,  # termination resistance (ohms)
    P_TPS=30e-3,  # thermal phase shifter power (W)
    
    P_PD=2e-3,  # photodetector power (W)
    
    P_TIA=44e-6, #107e-3,  # TIA power (W) 107mW for 60GHz TIA, 44µW for low speed integrator
    
    P_DAC_prime=423e-3,  # DAC power at reference rate (W)
    P_TX_prime=83e-3,  # transmitter DSP power at reference rate (W)
    P_ADC_prime=315.2e-3,  # ADC power at reference rate (W)
    P_RX_prime=185e-3,  # receiver DSP power at reference rate (W)

    
    f_Sa=256e9,  # sample rate (Hz)
    f_Sa_ref=112e9,  # reference sample rate for scaling (Hz)
    
    P_comb=1.0,  # comb laser power (W) for scalable system
    K=64,  # number of wavelengths
    M=64,  # number of spatial channels
    P_TPS_eff=5.6e-3, # efficient thermal phase shifter power per channel (W)
    
    P_sw=122e-9,  # switch power (W)

    f_B=106e9,  # baud rate (Hz)
):
    # laser power consumption
    P_L = V_L * ((P_o / gamma_L) + I_th)

    # modulator power consumption
    V_RMS = V_PP / (2*np.sqrt(2))
    P_mod = (V_RMS**2) / R + (1/2) * P_TPS
    P_mod_eff = (V_RMS**2) / R + (1/2) * P_TPS_eff

    # dac and transmitter power, scaled by sample rate
    P_DAC = P_DAC_prime * (f_Sa / f_Sa_ref)
    P_TX = (0.8**2) * P_TX_prime * (f_Sa / f_Sa_ref)

    # adc and receiver power, scaled by sample rate
    P_ADC = P_ADC_prime * (f_Sa / f_Sa_ref)
    P_RX = P_RX_prime * (f_Sa / f_Sa_ref)

    throughput = 2*K*M*f_B # OPS

    if K == 1 and M == 1:
        # total power consumption for single system
        P_T = (
            P_L + 2 * (P_mod + P_DAC + P_TX) + P_PD + P_TIA + P_ADC + P_RX
        )
    else:
        P_DAC=35e-15*f_Sa
        P_ADC=2.55e-3
        # scalable system: power with multiple wavelengths & spatial multiplexing
        P_T = (
            P_comb
            + (K + M) * (P_mod_eff + P_DAC + P_TX)
            + (K * M) * (P_PD + P_TIA + P_sw)
            + M * (P_ADC + P_RX)
        )

    compute_efficiency = throughput / P_T # OPS/W

    return P_T, compute_efficiency, throughput

# run with default parameters

KM = [1, 16, 32, 64, 128, 256]

efficiency = []
through = []

for km in KM:
    P_T, compute_efficiency, throughput = compute_power(K=km, M=km)
    print(f'K = {km}, M = {km} & {throughput/1e12:.3f} TOPS & {compute_efficiency/1e12:.3f} TOPS/W & {1e15/compute_efficiency:.3f} fJ/OP')

    efficiency.append(compute_efficiency)
    through.append(throughput)


# plot the results
import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot(KM, np.array(efficiency)/1e9, 'o-')
plt.xlabel('K = M')
plt.ylabel('Compute Efficiency (GOPS/W)')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(KM, np.array(through)/1e9, 'o-')
plt.xlabel('K = M')
plt.ylabel('Throughput (GOPS)')
plt.grid()

plt.tight_layout()
plt.show()
