import unittest
import os
from research_ai.agent import Agent
from research_ai.datastore import Datastore

class ValidatorTests(unittest.TestCase):
    def test_is_on_topic_positive(self):
        root = "Best 35mm lenses for Nikon Z mount"
        a = Agent(topic=root)
        self.assertTrue(a._is_on_topic("Nikon Z 35mm lens review", root))
        self.assertTrue(a._is_on_topic("Nikon Z 35mm f/1.8 S performance", root))

    def test_is_on_topic_negative(self):
        root = "Best 35mm lenses for Nikon Z mount"
        a = Agent(topic=root)
        self.assertFalse(a._is_on_topic("Canon RF 35mm lens review", root))
        self.assertFalse(a._is_on_topic("Canon RF mount adapter discussion", root))

    def test_derive_subtopics_avoids_generic(self):
        root = "Best 35mm lenses for Nikon Z mount"
        a = Agent(topic=root)
        summaries = [
            {"title": "Nikon Z-mount: overview", "summary": "Nikon Z-mount is an interchangeable lens mount developed by Nikon for mirrorless cameras."},
            {"title": "Popular 35mm options", "summary": "Many manufacturers make 35mm lenses; Nikon has fast primes and some third-party options."},
            {"title": "Canon RF mount notes", "summary": "Canon RF mount is Canon's system and unrelated to Nikon Z mount."},
        ]
        subs = a._derive_subtopics(root, summaries, max_children=3)
        # ensure we get at least one subtopic; any 'Canon' candidates should be filtered by _is_on_topic
        self.assertTrue(len(subs) >= 1)
        has_valid = False
        for s in subs:
            if 'Canon' in s:
                self.assertFalse(a._is_on_topic(s, root))
            else:
                if a._is_on_topic(s, root):
                    has_valid = True
        self.assertTrue(has_valid)

    def test_prompt_logging(self):
        db = Datastore('test_logs.db')
        db.init()
        pid = db.save_prompt_response(job_id='job1', task_id='t1', role='test', prompt_text='p', response_text='r', metadata={'k': 'v'})
        self.assertIsNotNone(pid)
        # query raw table
        cur = db.conn.cursor()
        cur.execute("SELECT id, job_id, task_id, role, prompt_text, response_text, metadata_json FROM prompt_logs WHERE id=?", (pid,))
        row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[1], 'job1')
        # cleanup
        try:
            os.remove('test_logs.db')
        except Exception:
            pass

if __name__ == '__main__':
    unittest.main()
