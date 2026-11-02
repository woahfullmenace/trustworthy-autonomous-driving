"""Apply explicit concept overrides to a saved validation prediction."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.interventions import load_inputs, manual_intervention, parse_overrides


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-id',required=True)
    parser.add_argument('--set',dest='overrides',action='append',required=True,metavar='CONCEPT=0|1')
    args=parser.parse_args()
    try:
        parse_overrides(args.overrides)
        result=manual_intervention(load_inputs(ROOT),args.sample_id,args.overrides)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.exit(1,f'Concept intervention failed: {exc}\n')
    print(json.dumps(result,indent=2))
