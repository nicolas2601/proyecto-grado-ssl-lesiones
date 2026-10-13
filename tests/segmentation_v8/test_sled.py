import unittest
import numpy as np
from dermssl.segmentation_methods.sled.entropy import structural_entropy,merge_and_refine
from dermssl.segmentation_methods.sled.pipeline import segment,SLEDConfig,ght_threshold
from dermssl.w3_components import DegenerateCase


class SLEDTest(unittest.TestCase):
    def test_entropy_matches_two_level_definition(self):
        a=np.array([[0,3,.1,.1],[3,0,.1,.1],[.1,.1,0,3],[.1,.1,3,0]])
        labels=np.array([0,0,1,1]);degree=a.sum(1);volume=a.sum();expected=0.
        for c in [0,1]:
            members=labels==c;v=degree[members].sum();boundary=a[np.ix_(members,~members)].sum()
            expected-=boundary/volume*np.log2(v/volume)
            expected-=np.sum(degree[members]/volume*np.log2(degree[members]/v))
        self.assertAlmostEqual(structural_entropy(a,labels),expected)
    def test_greedy_and_relocation_do_not_increase_entropy(self):
        rng=np.random.default_rng(4);a=rng.uniform(0,1,(20,20));a=(a+a.T)/2;np.fill_diagonal(a,0)
        labels,audit=merge_and_refine(a,max_refinement_passes=5)
        self.assertLessEqual(structural_entropy(a,labels),structural_entropy(a,np.arange(20))+1e-8)
        self.assertTrue(np.all(np.diff(audit['entropy_trace'])<=1e-8))
        again,_=merge_and_refine(a,max_refinement_passes=5);np.testing.assert_array_equal(labels,again)
    def test_end_to_end_synthetic_is_deterministic(self):
        yy,xx=np.indices((64,64));base=180+(xx//4)+(yy//5)
        image=np.stack((base,base-5,base-10),axis=-1).astype(np.uint8)
        lesion=((yy-32)/14)**2+((xx-32)/12)**2<1
        image[lesion]=np.stack((45+xx[lesion]//3,35+yy[lesion]//4,30+xx[lesion]//5),axis=-1)
        cfg=SLEDConfig(scales=(20,35),self_tuning_k=3,knn_k=5,trees=20,refinement_passes=3)
        one=segment(image,cfg=cfg);two=segment(image,cfg=cfg)
        np.testing.assert_array_equal(one['mask'],two['mask'])
        self.assertEqual(one['mask'].shape,(64,64));self.assertTrue(np.isfinite(one['probability']).all())
        self.assertTrue(one['mask'].any());self.assertFalse(one['mask'].all())
    def test_constant_and_zero_volume_abstain(self):
        with self.assertRaises(DegenerateCase):merge_and_refine(np.zeros((4,4)))
        with self.assertRaises(DegenerateCase):ght_threshold(np.ones(30))
        with self.assertRaises((DegenerateCase,ValueError)):segment(np.full((32,32,3),150,np.uint8),cfg=SLEDConfig(scales=(10,)))


if __name__=='__main__':unittest.main()
