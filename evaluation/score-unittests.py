import unittest
import sys

import numpy as np
import sklearn.metrics

from score_segmentations import get_overlaps, score_masks, addmajoritvote_mask, score_masks_macro, calculate_perinstance_agreement
from score_cvqa import get_accuracy_score, organize_values, calculate_accuracy


class TestScoreSegmentation(unittest.TestCase):

    def setUp( self ) :
        self.masks1 = [
            np.array([[0,1,0],[0,0,0],[1,1,1]]),
            np.array([[0,1,0],[0,0,0],[1,1,1]]),
            np.array([[0,1,1],[0,0,0],[0,1,1]]),
            np.array( [[0,1,1],[0,0,1],[0,1,1]])
        ]
        self.masks2 = [
            np.array([[0,1,0],[0,0,0],[1,1,1]]),
            np.array([[1,0,1],[1,1,1],[0,0,0]]),
            np.array([[1,0,1],[0,1,1],[0,0,0]]),
            np.array( [[1,0,1],[0,1,1],[0,0,0]])
        ]

    def test_getoverlaps(self):
        #exactly equals
        mask1 = self.masks1[0]
        mask2 = self.masks2[0]
        tp, fn, fp = get_overlaps( mask1, mask2 )
        self.assertEqual(tp,4)
        self.assertEqual(fn,0)
        self.assertEqual(fp,0)

        #exactly 0
        mask1 = self.masks1[1]
        mask2 = self.masks2[1]
        tp, fn, fp = get_overlaps( mask1, mask2 )
        self.assertEqual(tp,0)
        self.assertEqual(fn,4)
        self.assertEqual(fp,5)

        #somethine else
        mask1 = self.masks1[2]
        mask2 = self.masks2[2]
        tp, fn, fp = get_overlaps( mask1, mask2 )
        self.assertEqual(tp,1)
        self.assertEqual(fn,3)
        self.assertEqual(fp,3)

        #somethine else
        mask1 = self.masks1[3]
        mask2 = self.masks2[3]
        tp, fn, fp = get_overlaps( mask1, mask2 )
        self.assertEqual(tp,2)
        self.assertEqual(fn,3)
        self.assertEqual(fp,2)

    def test_scoremasks(self):

        imageid2_reffnormask ={
            'IMG_ENC01_01':self.masks1[0],
            'IMG_ENC02_01':self.masks1[1],
            'IMG_ENC03_01':self.masks1[2],
            'IMG_ENC04_01':self.masks1[3],
        }
        imageid2_predfnormask = {
            'IMG_ENC01_01':self.masks2[0],
            'IMG_ENC02_01':self.masks2[1],
            'IMG_ENC03_01':self.masks2[2],
            'IMG_ENC04_01':self.masks2[3],
        }
        instances = ['IMG_ENC01_01','IMG_ENC02_01','IMG_ENC03_01','IMG_ENC04_01']
        results = score_masks( imageid2_reffnormask, imageid2_predfnormask, instances )

        jaccard = 7/(7+10+10)
        dice = 2*7/(2*7+10+10)
        self.assertEqual( results['jaccard'], jaccard)
        self.assertEqual( results['dice'], dice)

    def test_getmajorityvote(self):
        imageid2fns_bylabeler ={
            'ann0': {
                'IMG_ENC01_01':np.array([[0,1,0,0],[0,0,0,1],[1,1,1,0]]),
                'IMG_ENC01_02': np.array([[0,1,0,0],[0,0,0,0],[0,0,0,0]]),
                'IMG_ENC01_03': np.array([[1,1,0,0],[0,0,1,0],[0,1,1,0]]),
            },
            'ann1': {
                'IMG_ENC01_01':np.array([[0,0,0,0],[1,0,0,1],[0,1,1,0]]),
                'IMG_ENC01_03': np.array([[1,1,0,0],[0,0,1,0],[0,1,0,0]]),
            },
            'ann2': {
                'IMG_ENC01_01':np.array([[1,1,1,1],[1,0,0,1],[0,1,1,0]]),
                'IMG_ENC01_02': np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1]]),
                'IMG_ENC01_03': np.array([[0,1,0,1],[0,0,1,0],[0,1,0,0]]),
            },
            'ann3': {
                'IMG_ENC01_03': np.array([[0,1,1,1],[0,1,1,0],[0,1,1,0]]),
                'IMG_ENC01_04': np.array([[0,1,1,1],[0,1,1,0],[0,1,1,0]]),
            },
        }
        annotators = ['ann0','ann1','ann2','ann3']
        imageids = ['IMG_ENC01_01','IMG_ENC01_02','IMG_ENC01_03','IMG_ENC01_04']

        addmajoritvote_mask( imageid2fns_bylabeler, annotators, imageids )
        self.assertTrue('majorityvote' in imageid2fns_bylabeler)

        TOL = 0.0000001

        image_enc01_01_mv = imageid2fns_bylabeler['ann0']['IMG_ENC01_01'] + imageid2fns_bylabeler['ann1']['IMG_ENC01_01'] + imageid2fns_bylabeler['ann2']['IMG_ENC01_01']
        image_enc01_01_mv = np.where( image_enc01_01_mv >= 2, 1, 0 )
        self.assertTrue( np.linalg.norm( imageid2fns_bylabeler['majorityvote']['IMG_ENC01_01'] - image_enc01_01_mv) <TOL )

        image_enc01_02_mv = imageid2fns_bylabeler['ann0']['IMG_ENC01_02'] + imageid2fns_bylabeler['ann2']['IMG_ENC01_02']
        image_enc01_02_mv = np.where( image_enc01_02_mv >= 1, 1, 0 )
        self.assertTrue( np.linalg.norm( imageid2fns_bylabeler['majorityvote']['IMG_ENC01_02']-image_enc01_02_mv) <TOL  )

        image_enc01_03_mv = imageid2fns_bylabeler['ann0']['IMG_ENC01_03'] + imageid2fns_bylabeler['ann1']['IMG_ENC01_03'] + imageid2fns_bylabeler['ann2']['IMG_ENC01_03'] + imageid2fns_bylabeler['ann3']['IMG_ENC01_03']
        image_enc01_03_mv = np.where( image_enc01_03_mv >= 2, 1, 0 )
        self.assertTrue( np.linalg.norm( imageid2fns_bylabeler['majorityvote']['IMG_ENC01_03']-image_enc01_03_mv) <TOL  )

        image_enc01_04_mv = imageid2fns_bylabeler['ann3']['IMG_ENC01_04']
        image_enc01_04_mv = np.where( image_enc01_04_mv >= 1, 1, 0 )
        self.assertTrue( np.linalg.norm( imageid2fns_bylabeler['majorityvote']['IMG_ENC01_04']-image_enc01_04_mv )<TOL  )

    def test_getperinstanceagreement(self):
        imageid2reffns_bylabeler ={
            'ann0': {
                'IMG_ENC01_01':np.array([[0,1,0,0],[0,0,0,1],[1,1,1,0]]),
                'IMG_ENC01_02': np.array([[0,1,0,0],[0,0,0,0],[0,0,0,0]]),
                'IMG_ENC01_03': np.array([[1,1,0,0],[0,0,1,0],[0,1,1,0]]),
            },
            'ann1': {
                'IMG_ENC01_01':np.array([[0,0,0,0],[1,0,0,1],[0,1,1,0]]),
                'IMG_ENC01_03': np.array([[1,1,0,0],[0,0,1,0],[0,1,0,0]]),
            },
            'ann2': {
                'IMG_ENC01_01':np.array([[1,1,1,1],[1,0,0,1],[0,1,1,0]]),
                'IMG_ENC01_02': np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1]]),
                'IMG_ENC01_03': np.array([[0,1,0,1],[0,0,1,0],[0,1,0,0]]),
            },
            'ann3': {
                'IMG_ENC01_03': np.array([[0,1,1,1],[0,1,1,0],[0,1,1,0]]),
                'IMG_ENC01_04': np.array([[0,1,1,1],[0,1,1,0],[0,1,1,0]]),
            },
        }
        labelers_gold = ['ann0','ann1','ann2','ann3']
        imageids = ['IMG_ENC01_01','IMG_ENC01_02','IMG_ENC01_03','IMG_ENC01_04']

        imageid2predfns ={
            'IMG_ENC01_01':np.array([[0,1,0,1],[0,1,0,0],[1,0,1,0]]),
            'IMG_ENC01_02': np.array([[0,1,0,0],[0,0,0,1],[0,1,0,0]]),
            'IMG_ENC01_03': np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0]]),
            'IMG_ENC01_04': np.array([[0,0,1,1],[0,1,0,0],[0,0,0,0]]),
        }

        jaccard_expected = [
            [ sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann0']['IMG_ENC01_01'].flatten(),imageid2predfns['IMG_ENC01_01'].flatten()),
              sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann1']['IMG_ENC01_01'].flatten(),imageid2predfns['IMG_ENC01_01'].flatten()),
              sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann2']['IMG_ENC01_01'].flatten(),imageid2predfns['IMG_ENC01_01'].flatten())
            ],
            [ sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann0']['IMG_ENC01_02'].flatten(),imageid2predfns['IMG_ENC01_02'].flatten()),
              sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann2']['IMG_ENC01_02'].flatten(),imageid2predfns['IMG_ENC01_02'].flatten()),
            ],
            [ sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann0']['IMG_ENC01_03'].flatten(),imageid2predfns['IMG_ENC01_03'].flatten()),
              sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann1']['IMG_ENC01_03'].flatten(),imageid2predfns['IMG_ENC01_03'].flatten()),
              sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann2']['IMG_ENC01_03'].flatten(),imageid2predfns['IMG_ENC01_03'].flatten()),
              sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann3']['IMG_ENC01_03'].flatten(),imageid2predfns['IMG_ENC01_03'].flatten()),
            ],
            [ 
              sklearn.metrics.jaccard_score(imageid2reffns_bylabeler['ann3']['IMG_ENC01_04'].flatten(),imageid2predfns['IMG_ENC01_04'].flatten()),
            ]
        ]
        dice_expected = [
            [ self.dice_coefficient(imageid2reffns_bylabeler['ann0']['IMG_ENC01_01'].flatten(),imageid2predfns['IMG_ENC01_01'].flatten()),
              self.dice_coefficient(imageid2reffns_bylabeler['ann1']['IMG_ENC01_01'].flatten(),imageid2predfns['IMG_ENC01_01'].flatten()),
              self.dice_coefficient(imageid2reffns_bylabeler['ann2']['IMG_ENC01_01'].flatten(),imageid2predfns['IMG_ENC01_01'].flatten())
            ],
            [ self.dice_coefficient(imageid2reffns_bylabeler['ann0']['IMG_ENC01_02'].flatten(),imageid2predfns['IMG_ENC01_02'].flatten()),
              self.dice_coefficient(imageid2reffns_bylabeler['ann2']['IMG_ENC01_02'].flatten(),imageid2predfns['IMG_ENC01_02'].flatten()),
            ],
            [ self.dice_coefficient(imageid2reffns_bylabeler['ann0']['IMG_ENC01_03'].flatten(),imageid2predfns['IMG_ENC01_03'].flatten()),
              self.dice_coefficient(imageid2reffns_bylabeler['ann1']['IMG_ENC01_03'].flatten(),imageid2predfns['IMG_ENC01_03'].flatten()),
              self.dice_coefficient(imageid2reffns_bylabeler['ann2']['IMG_ENC01_03'].flatten(),imageid2predfns['IMG_ENC01_03'].flatten()),
              self.dice_coefficient(imageid2reffns_bylabeler['ann3']['IMG_ENC01_03'].flatten(),imageid2predfns['IMG_ENC01_03'].flatten()),
            ],
            [ 
              self.dice_coefficient(imageid2reffns_bylabeler['ann3']['IMG_ENC01_04'].flatten(),imageid2predfns['IMG_ENC01_04'].flatten()),
            ]
        ]

        data_jaccard, data_dice = calculate_perinstance_agreement( imageid2reffns_bylabeler, labelers_gold, imageid2predfns, imageids )

        #check sizes are correct
        self.assertEqual( len(data_jaccard), len(jaccard_expected))
        self.assertEqual( len(data_dice), len(dice_expected) )

        for i in range(len(jaccard_expected)):
            self.assertEqual( len(data_jaccard[i]), len(jaccard_expected[i]) )
            self.assertEqual( len(data_dice[i]), len(dice_expected[i]) )

            for j, x in enumerate( jaccard_expected[i] ):
                self.assertEqual( x, data_jaccard[i][j] )
                self.assertEqual( dice_expected[i][j], data_dice[i][j] )
        
    def dice_coefficient( self, x, y ):
        intersection = np.multiply(x,y)
        tp = np.sum( intersection )
        return 2*tp/(np.sum(x)+np.sum(y))

    def test_getmacro(self):
        data_jacc = [
            [0.1,0.3,0.4],
            [0.2, 1.0],
            [0.9, 0.8, 0.6],
            [0.6]
        ]
        data_dice = [
            [0.1,0.4,0.2],
            [0.2, 0.9],
            [0.7, 0.5,0.6],
            [0.8]
        ]
        results = score_masks_macro( data_jacc, data_dice ) 

        jaccard_meanofmax = np.mean([0.4,1.0,0.9,0.6])
        self.assertEquals( results['jaccard_meanofmax'], jaccard_meanofmax)

        jaccard_meanofmean = np.mean([np.mean([0.1,0.3,0.4]),
                                     np.mean([0.2, 1.0]),
                                     np.mean([0.9, 0.8, 0.6]),
                                     np.mean([0.6]),
                                     ])
        self.assertEquals( results['jaccard_meanofmean'], jaccard_meanofmean)

        dice_meanofmax = np.mean([0.4,0.9,0.7,0.8])
        self.assertEquals( results['dice_meanofmax'], dice_meanofmax)

        dice_meanofmean = np.mean([np.mean([0.1,0.4,0.2]),
                                     np.mean([0.2, 0.9]),
                                     np.mean([0.7, 0.5,0.6]),
                                     np.mean([0.8]),
                                     ])
        self.assertEquals( results['dice_meanofmean'], dice_meanofmean)


class TestCVQA(unittest.TestCase):

    def test_accuracy_score( self ) :
        gold_items = [ ['a','b','b'],
                       ['c'],
                       ['d','e','f','g'],
                     ]
        sys_items = [ ['a','b','c'],
                       ['c','d'],
                       ['d','e'],
                     ]
        acc = get_accuracy_score( gold_items, sys_items )

        accuracy_expected = (2/3 + 1/2 + 2/4)/3
        self.assertEqual( acc, accuracy_expected )

        gold_items = [ ['b','b'],
                       ['c','g','e','f'],
                       ['f','g'],
                       ['e']
                     ]
        sys_items = [ ['a','b','c'],
                       ['c','e','e','c'],
                       ['d','e','g'],
                       []
                     ]
        acc = get_accuracy_score( gold_items, sys_items )

        accuracy_expected = (1/3 + 2/4 + 1/3 + 0 )/4
        self.assertEqual( acc, accuracy_expected )

    def test_organize_values(self):
        data = [
            {
                "encounter_id": "ENC001",
                "CQID010-001": 'yes',
                "CQID011-001": 'no',
                "CQID011-002": '20',
                "CQID011-003": 'fingers',
                "CQID012-001": 'yes',
                "CQID012-002": 'no'
            },
            {
                "encounter_id": "ENC002",
                "CQID010-001": 'no',
                "CQID011-001": 'yes',
                "CQID011-002": '50',
                "CQID011-003": 'leg',
                "CQID012-001": 'no',
                "CQID012-002": 'no'
            },
            {
                "encounter_id": "ENC003",
                "CQID010-001": 'yes',
                "CQID011-001": 'yes',
                "CQID011-002": '31',
                "CQID011-003": 'back',
                "CQID012-001": 'no',
                "CQID012-002": 'yes'
            }
        ]

        qid2val_byencounterid_expected ={
            "ENC001":{
                "CQID010": ['yes'],
                "CQID011": ['no','20','fingers'],
                "CQID012": ['yes','no']
            },
            "ENC002":{
                "CQID010": ['no'],
                "CQID011": ['yes','50','leg'],
                "CQID012": ['no','no']
            },
            "ENC003":{
                "CQID010": ['yes'],
                "CQID011": ['yes','31','back'],
                "CQID012": ['no','yes']
            },
        }

        qid2val_byencounterid = organize_values(data)
        
        for encounter_id in ['ENC001','ENC002','ENC003'] :
            for qid in ['CQID010','CQID011','CQID012']:
                # print('{}\t{}:\n{}\n{}'.format(encounter_id,qid,
                #                                 str( qid2val_byencounterid[encounter_id][qid]),
                #                                 str(qid2val_byencounterid_expected[encounter_id][qid])
                #                                 ), file=sys.stderr)
                self.assertListEqual( qid2val_byencounterid[encounter_id][qid],
                                      qid2val_byencounterid_expected[encounter_id][qid]
                                    )

    def test_categoryaccuracy(self):
        qid2val_byencounterid_gold ={
            "ENC001":{
                "CQID010": ['yes'],
                "CQID011": ['no','20','fingers'],
                "CQID012": ['yes','no']
            },
            "ENC002":{
                "CQID010": ['no'],
                "CQID011": ['yes','50','leg','foot'],
                "CQID012": ['no','no']
            },
            "ENC003":{
                "CQID010": ['yes'],
                "CQID011": ['yes','31','back'],
                "CQID012": ['no','yes']
            },
        }
        qid2val_byencounterid_sys ={
            "ENC001":{
                "CQID010": ['yes'],
                "CQID011": ['no','20','fingers'],
                "CQID012": ['yes']
            },
            "ENC002":{
                "CQID010": ['no'],
                "CQID011": ['yes','50','leg'],
                "CQID012": ['no','no']
            },
            "ENC003":{
                "CQID010": ['yes'],
                "CQID011": ['no','32','back'],
                "CQID012": ['no','yes']
            },
        }
        qidparents = ['CQID010','CQID011','CQID012']

        results = calculate_accuracy( qid2val_byencounterid_gold, qid2val_byencounterid_sys, qidparents=qidparents)

        self.assertTrue( 'accuracy_CQID010' in results )
        self.assertTrue( 'accuracy_CQID011' in results )
        self.assertTrue( 'accuracy_CQID012' in results )
        self.assertTrue( 'accuracy_all' in results )

        self.assertEqual( results['accuracy_CQID010'], 1.0 )
        self.assertEqual( results['accuracy_CQID011'], (1 + 3/4 + 1/3 ) / 3 )
        self.assertEqual( results['accuracy_CQID012'], (1/2 + 1 + 1 ) /3 )
        self.assertEqual( results['accuracy_all'], (3 + 1 + 3/4 + 1/3 + 1/2 + 1 + 1 ) /9 )


if __name__ == '__main__':
    unittest.main()