
data = {}

# key = (run-start,run-end,pedrun)
data[(124396,124396,124397)] = "cygno04_sideA_exp7s_VGEM_460_VD_940"
data[(124398,124398,124397)] = "cygno04_sideA_exp7s_VGEM_450_VD_940"
data[(124400,124400,124399)] = "cygno04_sideA_exp7s_VGEM_440_VD_940"

data[(124402,124402,124401)] = "cygno04_sideA_exp7s_VGEM_460_VD_800"
data[(124404,124404,124403)] = "cygno04_sideA_exp7s_VGEM_450_VD_800"
data[(124406,124406,124405)] = "cygno04_sideA_exp7s_VGEM_440_VD_800"

data[(124408,124408,124407)] = "cygno04_sideA_exp7s_VGEM_460_VD_600"
data[(124410,124410,124409)] = "cygno04_sideA_exp7s_VGEM_450_VD_600"
data[(124412,124412,124411)] = "cygno04_sideA_exp7s_VGEM_440_VD_600"

for k,v in data.items():
    rstart,rstop,pedrun=k
    print(f"===> Now compute map for {v}, from run-range: [{rstart} - {rstop}] using pedesta run {pedrun}")
    cmd = f"python make_2dmaps_frommidas.py --run-start {rstart} --run-end {rstop} --pedrun {pedrun} --output {v}.root"
    print(f"\t{cmd}")
    #os.system(cmd)
    zmax = 55 if "VGEM_460" in v else (40 if "VGEM_450" in v else 25)
    plotcmd = f"python plot_2dmaps.py --input {v}.root --output-dir cernplots --output-name {v} --zmax-prof {zmax}"
    print(f"\t{plotcmd}")
    
    
