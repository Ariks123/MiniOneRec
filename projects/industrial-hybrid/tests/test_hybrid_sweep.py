import unittest
import random
import numpy as np

from hybrid.sweep import Catalog,Packed,merge_selected,alpha_groups,make_config,accumulate,rank_key


class ReplayTest(unittest.TestCase):
    def fixture(self):
        rng=random.Random(42)
        sids=[str(x) for x in range(65)]+["0","3","3","4"]
        catalog=Catalog.from_dict({"sid_by_item":sids,"target_freq":[rng.randrange(10) for _ in sids],
                                   "cold":[x%3==0 for x,_ in enumerate(sids)]})
        rows=[]
        for i in range(16):
            perm=rng.sample(range(len(sids)),len(sids))
            first=rng.randrange(11)
            second=rng.randrange(11)
            third=rng.randrange(11)
            scores={str(j):rng.choice([-3,-2,-2,-1]) for j in range(len(sids))}
            rounds=[perm[:first],perm[first:first+second],perm[first+second:first+second+third]]
            beam=rng.sample(list(catalog.aliases),50)
            rows.append(dict(row_id=i,target=i,beam=beam,scores=scores,
                             drafts={"ann":{"5":{"first":rounds[0],"5":rounds[1:]}}}))
        return catalog,rows

    def test_vectorized_replay_matches_online_reference(self):
        catalog,rows=self.fixture()
        block=Packed(rows,catalog)
        prim=block.items_in("ann",5,5)
        for t in (None,-2.5,-2,-1):
            state=np.cumsum((prim[0]>(-np.inf if t is None else t)).sum(axis=2),axis=1)
            for q in range(1,6):
                c=make_config("ann",5,q,5,t,[q/5]); out=block.replay(c,prim,state)
                for j,row in enumerate(rows):
                    accepted=[]
                    data=row["drafts"]["ann"]["5"]
                    for ordinal,g in enumerate([data["first"],*data["5"]],1):
                        accepted.extend((x,ordinal) for x in g if t is None or row["scores"][str(x)]>t)
                        if len(accepted)>=q: break
                    expected=merge_selected(accepted,row["beam"],row["scores"],catalog,5,q)
                    self.assertEqual(out[0][j].tolist(),[x.item_id for x in expected])
                    self.assertEqual(out[1][j].tolist(),[x.round for x in expected])
                    self.assertAlmostEqual(out[3][j],sum(x.source=="retrieval" for x in expected))
                    for sc,x in zip(out[2][j],expected):
                        if x.score is None:self.assertTrue(np.isnan(sc))
                        else:self.assertEqual(sc,x.score)

    def test_routing_boundaries(self):
        catalog,_=self.fixture()
        self.assertIn(.4,alpha_groups(1)[0])
        self.assertIn(.4,alpha_groups(10)[4])
        self.assertIn(.02,alpha_groups(50)[1])
        self.assertEqual(sum(map(len,alpha_groups(50).values())),12)
        canonical=catalog.aliases["3"][0]
        other=catalog.aliases["3"][1]
        out=merge_selected([(canonical,1)],["3"],{canonical:-2,other:-1},catalog,2,2)
        self.assertEqual([x.item_id for x in out],[other,canonical])
        self.assertEqual([x.source for x in out],["beam","retrieval"])
        self.assertEqual(merge_selected([],["3","0","4"],{},catalog,2,0)[0].item_id,canonical)

    def test_exact_and_sid_hit_differ_and_report_same_denominator(self):
        catalog,rows=self.fixture()
        row=rows[0]
        canonical=catalog.aliases["3"][0]
        row["beam"]=["3"]+[s for s in catalog.aliases if s!="3"][:49]
        row["target"]=catalog.aliases["3"][1]
        packed=Packed([row],catalog)
        counts=np.zeros((3,11))
        accumulate(packed,packed.pure(1),counts)
        self.assertNotEqual(row["target"],canonical)
        self.assertEqual(counts[0,:5].tolist(),[1,0,0,1,1])


if __name__=="__main__":
    unittest.main()
