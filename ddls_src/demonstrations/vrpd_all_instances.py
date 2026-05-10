from ddls_src.demonstrations.howto_vrpd_demonstration import run_matrix_scenario_demo

instances = ["A-n32-k5.vrp","A-n33-k5.vrp","A-n33-k6.vrp","A-n34-k5.vrp","A-n36-k5.vrp","A-n37-k5.vrp","A-n37-k6.vrp",
             "A-n38-k5.vrp","A-n39-k5.vrp","A-n44-k6.vrp","A-n45-k6.vrp","A-n45-k7.vrp","A-n46-k7.vrp","A-n48-k7.vrp",
             "A-n53-k7.vrp","A-n54-k7.vrp","A-n55-k9.vrp","A-n60-k9.vrp","A-n61-k9.vrp","A-n62-k8.vrp","A-n63-k9.vrp",
             "A-n63-k10.vrp","A-n64-k9.vrp","A-n65-k9.vrp","A-n69-k9.vrp","A-n80-k10.vrp"]

for instance in instances:
    run_matrix_scenario_demo(instance, 4, 4)