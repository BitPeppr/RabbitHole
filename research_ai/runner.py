#!/usr/bin/env python3
"""Single-file runner that wraps the existing orchestrator to run a single job.

Usage: python -m research_ai.runner --topic "Best 35mm lenses for Nikon Z mount"
"""
import argparse
import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) or '.')
from research_ai.datastore import Datastore
from research_ai.llm import LLM
from research_ai.web_search import WebSearchConnector
from research_ai.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', required=True)
    parser.add_argument('--output', default='research_report.md')
    args = parser.parse_args()

    ds = Datastore('research.db')
    ds.init()
    llm = LLM()
    conn = WebSearchConnector()
    orch = Orchestrator(datastore=ds, llm=llm, connector=conn, concurrency=1)
    print('Starting job...')
    asyncio.run(orch.run_job(topic=args.topic, max_depth=2, max_children=3, output_path=args.output))
    print('Done. Report at', os.path.abspath(args.output))

if __name__ == '__main__':
    main()
