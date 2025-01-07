import unittest

from synful import synapse


class TestSynapse(unittest.TestCase):
    def test_cluster_synapses(self):
        syn_1 = synapse.Synapse(id=1, location_pre=(1, 2, 3),
                                location_post=(10, 10, 0), id_segm_pre=1,
                                id_segm_post=10)
        syn_2 = synapse.Synapse(id=2, location_pre=(3, 4, 5),
                                location_post=(12, 14, 0), id_segm_pre=1,
                                id_segm_post=10)

        syn_3 = synapse.Synapse(id=3, location_pre=(0, 0, 0),
                                location_post=(30, 30, 0), id_segm_pre=1,
                                id_segm_post=10)
        syn_4 = synapse.Synapse(id=4, location_pre=(0, 0, 0),
                                location_post=(32, 32, 0), id_segm_pre=1,
                                id_segm_post=10)

        syn_5 = synapse.Synapse(id=5, location_pre=(0, 0, 0),
                                location_post=(10, 10, 0), id_segm_pre=1,
                                id_segm_post=5)

        synapses, removed_ids = synapse.cluster_synapses(
            [syn_1, syn_2, syn_3, syn_4, syn_5],
            5)
        ids = [syn.id for syn in synapses]

        self.assertTrue(1 in ids)
        self.assertTrue(3 in ids)
        self.assertTrue(5 in ids)
        self.assertFalse(2 in ids)
        self.assertFalse(4 in ids)

        self.assertEqual(tuple(synapses[0].location_post), (11, 12, 0))
        self.assertEqual(tuple(synapses[0].location_pre), (2, 3, 4))

        self.assertTrue(2 in removed_ids)


if __name__ == '__main__':
    unittest.main()
