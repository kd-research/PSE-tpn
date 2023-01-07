from data.steersim_quest import steersim_call_parallel


def initial_sample_steersim():
    import os
    import shutil
    import numpy as np
    from dotenv import load_dotenv

    load_dotenv(verbose=True)
    #    shutil.rmtree(os.getenv("SteersimRecordPath"), ignore_errors=True)
    #    os.makedirs(os.getenv("SteersimRecordPath"), exist_ok=True)
    num_parameters = 9
    numbers = np.random.uniform(0, 1, (4000, num_parameters))
    steersim_call_parallel(numbers)

    numbers = np.random.uniform(0, 1, (1000, num_parameters))
    steersim_call_parallel(numbers, generate_for_testcases=True)

if True:
    initial_sample_steersim()
