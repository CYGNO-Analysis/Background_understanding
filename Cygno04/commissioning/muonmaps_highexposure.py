import os

data = {}

# key = (run-start,run-end,pedrun)
# data[(124396,124396,124397)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_940"
# data[(124398,124398,124397)] = "cygno04_sideA_exp7s_nostdcut_VGEM_450_VD_940"
# data[(124400,124400,124399)] = "cygno04_sideA_exp7s_nostdcut_VGEM_440_VD_940"

# data[(124402,124402,124401)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_800"
# data[(124404,124404,124403)] = "cygno04_sideA_exp7s_nostdcut_VGEM_450_VD_800"
# data[(124406,124406,124405)] = "cygno04_sideA_exp7s_nostdcut_VGEM_440_VD_800"

# data[(124408,124408,124407)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_600"
# data[(124410,124410,124409)] = "cygno04_sideA_exp7s_nostdcut_VGEM_450_VD_600"
# data[(124412,124412,124411)] = "cygno04_sideA_exp7s_nostdcut_VGEM_440_VD_600"


data[(124650,124650,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_000"
data[(124649,124649,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_040"
data[(124648,124648,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_140"
data[(124647,124647,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_240"
data[(124646,124646,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_340"
data[(124645,124645,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_440"
data[(124644,124644,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_540"
data[(124643,124643,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_640"
data[(124642,124642,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_740"
data[(124641,124641,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_840"
data[(124643,124643,124639)] = "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_940"


for k,v in data.items():
    rstart,rstop,pedrun=k
    print(f"===> Now compute map for {v}, from run-range: [{rstart} - {rstop}] using pedesta run {pedrun}")
    cmd = f"python make_2dmaps_frommidas.py --nsigma -100.0 --run-start {rstart} --run-end {rstop} --pedrun {pedrun} --output {v}.root"
    print(f"\t{cmd}")
    os.system(cmd)
    
    #zmax = 55 if "VGEM_460" in v else (40 if "VGEM_450" in v else 25)
    plotcmd = f"python plot_2dmaps.py --input {v}.root --output-dir cernplots --output-name {v} --zmax-prof 55"
    print(f"\t{plotcmd}")
    os.system(plotcmd)
    
    
