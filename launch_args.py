import os
import time
import argparse

# ResultDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

# def check_file(domain, level, amount, algo):
#     ''' check if the result is already exist '''
#     json_file = f'{domain}-{level}-{amount},{algo}.json'
#     return json_file in os.listdir(ResultDir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='select from `cstr`, `winding`, `thickener`')
    parser.add_argument('--sp', type=str, default=None, help='select from `0.25`, `0.5`, `1`')
    parser.add_argument('--algo', type=str, default='ode', help='select from `ode`, `sde`')
    parser.add_argument('--evenly', type=str, default=None, help='evenly')
    args = parser.parse_args()

    # if not os.path.exists(ResultDir): os.makedirs(ResultDir)

    if args.algo is None:
        algos = ['ode', 'sde']
    else:
        algos = [args.algo]

    for algo in algos:
        if args.data is None:
            datas = ['cstr', 'winding', 'thickener']
        else:
            datas = [args.data]
        for data in datas:
            if args.sp is None:
                data_sp = ['0.25', '0.5', '1']
            else:
                data_sp = [args.sp]
            for sp in data_sp:
                if args.evenly is None:
                    data_evenly = ['False', 'True']
                else:
                    data_evenly = [args.amount]
                for evenly in data_evenly:
                    if algo == 'ode':
                        # --train-dir ode_save/thickener --batch-size 1024 --data thickener --sp=0.25 --evenly=False
                        os.system(f'python main.py --train-dir ode_save/{data} --batch-size 1024 --data {data} --sp {sp} --evenly {evenly}')
                    elif algo == 'sde':
                        os.system(f'python latent_sde.py --train-dir sde_save/{data} --batch-size 1024 --show-prior False --data {data} --adjoint False --dt 5e-2 --adjoint False --sp {sp} --evenly {evenly}')
                    time.sleep(20)

