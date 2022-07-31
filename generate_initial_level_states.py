import argparse
from pathlib import Path
import numpy as np
from gym_pcgrl.envs.reps.representation import Representation
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map


def build_level(prob_name):
    rep = Representation()
    prob = PROBLEMS[prob_name]()
    tile_probs = get_int_prob(prob._prob, prob.get_tile_types())
    rep.reset(prob._width, prob._height, tile_probs)
    return rep._map

def save_level(level, prob, level_id):
    # create directory structure
    # ./{prob}_levels/{1}/level.txt
    output_path = Path(f'{prob}_levels')
    output_path.mkdir(exist_ok=True)
    with open(Path(output_path, f'level_{level_id}.txt'), 'w+') as f:
        np.savetxt(f, level)
    

def main(prob, n_levels=40):
    for i in range(n_levels):
        level = build_level(prob)
        save_level(level, prob, str(i))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', dest='prob', type=str, required=True)
    parser.add_argument('-n', '--n_levels', dest='n_levels', type=int, default=40)
    args = parser.parse_args()
    main(args.prob, args.n_levels)





