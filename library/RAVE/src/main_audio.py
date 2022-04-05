import os
import glob

from multiprocessing import Process, Queue

from RAVE.audio.AudioManager import AudioManager


def run_loop(file_queue, worker_num):

    # TODO: ADD POSSIBILITY TO RESET AUDIO_MANAGER INSTEAD OF RECREATING
    while not file_queue.empty():

        # Get file in queue
        audio_file = file_queue.get()
        print(f'Worker {worker_num}: Starting speech enhancement on {audio_file}')

        # Run audio manager
        audio_manager = AudioManager(debug=False, mask=False, use_timers=False)
        audio_dict = {
            'name': 'loop_sim_source',
            'type': 'sim',
            'file': audio_file
        }
        audio_manager.initialise_audio(source=audio_dict)
        audio_manager.main_loop()
        # audio_manager.reset_manager()


def main2(LOOP_DIR=None):

    # Get all files in a subdirectory
    if LOOP_DIR:
        worker_count = 3
        worker_list = []
        file_queue = Queue()
        # Get audio files
        input_files = glob.glob(os.path.join(LOOP_DIR, '**/audio.wav'))
        for audio_file in input_files:
            file_queue.put(audio_file)

        # Start workers
        for w in range(1, worker_count+1):
            p = Process(target=run_loop, args=(file_queue, w, ))
            worker_list.append(p)
            p.start()
            print(f'Worker {w}: started.')

        # Join workers when done
        for w_num, p in enumerate(worker_list):
            p.join()
            print(f'Worker {w_num+1}: finished.')
    else:
        audio_manager = AudioManager(debug=True, mask=False, use_timers=False)
        audio_manager.initialise_audio()
        audio_manager.main_loop()


if __name__ == "__main__":
    # main2()
    main2(LOOP_DIR='C:\\GitProjet\\pipeline\\new')
