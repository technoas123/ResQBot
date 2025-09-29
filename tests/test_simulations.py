import unittest
from src.utils.simulation import DisasterSceneSimulator

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.simulator = DisasterSceneSimulator({})
        
    def test_scene_generation(self):
        scene, has_human = self.simulator.generate_scene(with_human=True)
        self.assertIsNotNone(scene)
        self.assertTrue(has_human)
        
        scene, has_human = self.simulator.generate_scene(with_human=False)
        self.assertIsNotNone(scene)
        self.assertFalse(has_human)

if __name__ == '__main__':
    unittest.main()