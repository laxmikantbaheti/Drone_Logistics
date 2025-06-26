from mlpro.bf.ml import Scenario


class VRPDScenario(Scenario):

    def __init__(self):

        Scenario.__init__(self)



    def _run_cycle(self):

        # 1. Update the action masks for the system

        # 2. Check the order manager for new generated orders

        # 3. Check the updated action masks

        # 4. If there are actions available trigger particular agents to calculate the next decisions

        # 5. Simulate the system for next timestep

        pass


    def _reset(self, p_seed):

        pass
