import sys
import argparse

help_format = lambda arg: f'{arg[2]} (default: {arg[1]}).'

def parser(arg_specs: dict[str, tuple]):
    parser = argparse.ArgumentParser()

    for arg, arg_type in arg_specs.items():
        help = help_format(arg_type)
        if arg_type[0] == bool:
            parser.add_argument(f'--{arg}', action=argparse.BooleanOptionalAction, help=help, default=arg_type[1])
        else:
            parser.add_argument(f'--{arg}', type=arg_type[0], help=help, default=arg_type[1])

    args = parser.parse_args()
    return vars(args)

def parse_llm_args():
    arg_specs = {
        "mode": (int, 1, "Mode of llm operation: 1 for simple question, 2 for variety of questions, 3 for alternative responses, 4 embedding similarity."),
        "log": (bool, False, "Log the output of the models."),
    }

    if '--help' in sys.argv or '-h' in sys.argv:
        print('Testing LLM capabilities with different chat histories and prompts.')
        print('--Usage: python llm.py [options]--\n')
        for arg, arg_type in arg_specs.items():
            print(f'\t{arg}: \t{help_format(arg_type)}')
        print('\thelp: Print this message.')
        return None
    
    

    return parser(arg_specs)

final_args = None
def parse_args():
    # Store and return parsed arguments 
    global final_args
    if final_args is not None:
        return final_args

    # Else parse the arguments
    final_args = parse_llm_args()
    if final_args is None:
        sys.exit(0)
    return final_args
