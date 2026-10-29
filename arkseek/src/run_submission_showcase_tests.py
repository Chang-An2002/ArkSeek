"""Submission showcase regressions: fake network, no .env or real data access."""
import unittest
from unittest.mock import patch

with patch('dotenv.load_dotenv',return_value=False):
    from conversation_memory import ConversationSession

class CargoDisplayTests(unittest.TestCase):
    def test_original_names_and_special_mode_scope_are_preserved(self):
        rows=[{'name':'能天使'},{'name':'Stormeye(卫戍协议)'}]
        with patch('conversation_memory.query_operators',return_value=rows) as query:
            answer=ConversationSession._try_answer_from_cargo_filter(object(),'狙击','6')
        query.assert_called_once_with(profession='狙击',rarity=6)
        self.assertIn('共2个：能天使、Stormeye(卫戍协议)',answer.core)
        self.assertIn('未排除特殊模式专属干员',answer.display)
        self.assertIn('https://prts.wiki/w/干员一览',answer.display)
        for bad in ['.py','不存在编造','大概率','口径差异']:
            self.assertNotIn(bad,answer.display)

    def test_empty_result_is_distinct_from_unavailable_source(self):
        with patch('conversation_memory.query_operators',return_value=[]):
            answer=ConversationSession._try_answer_from_cargo_filter(object(),'狙击','6')
        self.assertTrue(answer.ok)
        self.assertIn('没有查到符合条件',answer.core)
        self.assertIn('来源：',answer.display)
        with patch('conversation_memory.query_operators',return_value=None):
            self.assertIsNone(ConversationSession._try_answer_from_cargo_filter(object(),'狙击','6'))

if __name__=='__main__':unittest.main(verbosity=2)
