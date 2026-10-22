import copy
import json
from pathlib import Path
import tempfile
import unittest

from astra_followup_package import (synthesis_request,classification_requests,
    validate_codebook,validate_assignment,freeze_packet)


def case(cid, split, name):
    return {'case_id':cid,'source_split':split,'records':{'left':{'name':name},'right':{'name':name+' edition'}},
        'gold_label':0,'prediction':1,'error_type':'FP','models':['NEVER_SEND_MODEL_ID'],'pair_id':'PRIVATE_PAIR_ID'}


def codebook():
    return {'categories':[{'code':'C01','name':'Synthetic category','definition':'Synthetic example only',
        'include_when':'Explicit edition difference','exclude_when':'No edition text',
        'supporting_discovery_case_ids':['d1']}],'overlap_policy':'Multiple supported categories allowed',
        'uncertainty_policy':'Use unclear or not_covered where appropriate'}


class Followup(unittest.TestCase):
    def setUp(self):
        self.discovery=[case('d1','prompt','Train item')]
        self.evaluation=[case('e1','test','Test item')]
        self.obs={'discover-d1':{'available':True,'observations':{'observations':[],'uncertainty':''}}}

    def test_synthesis_only_discovery_and_no_model_identity(self):
        row=synthesis_request(self.discovery,self.obs)
        body=json.dumps(row['body'])
        for prohibited in ['NEVER_SEND_MODEL_ID','PRIVATE_PAIR_ID','Test item']:
            self.assertNotIn(prohibited,body)
        with self.assertRaises(ValueError): synthesis_request(self.evaluation,{'discover-e1':self.obs['discover-d1']})
        self.obs['discover-d1']['available']=False
        with self.assertRaises(ValueError): synthesis_request(self.discovery,self.obs)

    def test_codebook_rejects_test_references_duplicates(self):
        b=codebook(); b['categories'][0]['supporting_discovery_case_ids']=['e1']
        with self.assertRaises(ValueError): validate_codebook(b,['d1'])
        b=codebook();b['categories']*=2
        with self.assertRaises(ValueError): validate_codebook(b,['d1'])

    def test_three_requests_fresh_ids_same_frozen_input(self):
        rows=classification_requests(self.evaluation,self.discovery,codebook())
        self.assertEqual(len(rows),3)
        self.assertEqual(len({r['custom_id'] for r in rows}),3)
        self.assertTrue(all(r['body']==rows[0]['body'] for r in rows))
        self.assertNotIn('NEVER_SEND_MODEL_ID',json.dumps(rows))
        self.assertEqual(rows[0]['body']['max_output_tokens'],2048)

    def test_mirrored_discovery_rejected(self):
        item=copy.deepcopy(self.discovery[0]);item.update(case_id='e1',source_split='test')
        item['records']={'left':item['records']['right'],'right':item['records']['left']}
        with self.assertRaises(ValueError): classification_requests([item],self.discovery,codebook())

    def test_evidence_and_status(self):
        result={'status':'categorized','assignments':[{'code':'C01','attribute':'name','left_evidence':'Test item',
            'right_evidence':'edition','description':'Synthetic observation'}],'uncertainty':''}
        self.assertFalse(validate_assignment(result,self.evaluation[0],['C01'])['internal_cause_validated'])
        bad=copy.deepcopy(result);bad['assignments'][0]['left_evidence']='fabricated'
        with self.assertRaises(ValueError): validate_assignment(bad,self.evaluation[0],['C01'])
        bad=copy.deepcopy(result);bad['status']='unclear'
        with self.assertRaises(ValueError): validate_assignment(bad,self.evaluation[0],['C01'])
        bad=copy.deepcopy(result);bad['assignments'][0].update(attribute='invented',left_evidence='',right_evidence='')
        with self.assertRaises(ValueError): validate_assignment(bad,self.evaluation[0],['C01'])

    def test_packet_is_unapproved_and_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'packet'
            row=synthesis_request(self.discovery,self.obs)
            manifest=freeze_packet(path,[row],{'synthetic_test_only':True})
            self.assertFalse(manifest['covered_by_discovery_USD11_approval'])
            self.assertFalse(manifest['submitted'])
            with self.assertRaises(FileExistsError): freeze_packet(path,[row],{})


if __name__=='__main__': unittest.main()
