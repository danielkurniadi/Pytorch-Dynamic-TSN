import os
import argparse
from pathlib import Path
from itertools import product
from multiprocessing import Pool, current_process

# core libs
import cv2
import numpy as np


vid_exts = ['.avi', '.mp4', '.webm', '.mov', '.mkv']


class Buffer():
    def __init__(self, size):
        self.size = size
        self.container = []
        self.batch_count = 0

    def enqueue(self, item):
        if len(self.container) < self.size:
            self.container.append(item)
        else:
            print('Buffer full')

    def dequeue(self):
        if not self.isempty():
            self.container.pop(0)
        else:
            print("Buffer empty")

    def clear(self):
        container = self.container
        self.container = []
        self.batch_count += 1
        return np.array(container)

    def get(self):
        return np.array(self.container)

    def isempty(self):
        return len(self.container) == 0

    def isfull(self):
        return (len(self.container) == self.size)


def search_videos_recursively(video_dir):
    exts = list(map(lambda x: '**/*'+x, vid_exts))
    files = []
    for ext in exts:
        files.extend(Path(video_dir).glob(ext))

    abspaths = map(os.path.abspath, files)
    return list(abspaths)


def cvApproxRankPooling_DIN(imgs):
    T = len(imgs)
  
    harmonics = []
    harmonic = 0
    for t in range(0, T+1):
        harmonics.append(harmonic)
        harmonic += float(1)/(t+1)

    weights = []
    for t in range(1 ,T+1):
        weight = 2 * (T - t + 1) - (T+1) * (harmonics[T] - harmonics[t-1])
        weights.append(weight)
        
    feature_vectors = []
    for i in range(len(weights)):
        feature_vectors.append(imgs[i] * weights[i])

    feature_vectors = np.array(feature_vectors)

    rank_pooled = np.sum(feature_vectors, axis=0)
    rank_pooled = cv2.normalize(rank_pooled, None, alpha=0, beta=255, 
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return rank_pooled


def run(vid_url, outname):
    """Main function.
    """
    current = current_process()
    cap = cv2.VideoCapture(vid_url)

    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    buffer = Buffer(10)
    success = True

    while success:
        success, frame = cap.read()

        if buffer.isfull():
            frames = buffer.clear()
            rank_pooled = cvApproxRankPooling_DIN(frames)
            
            outpath = "{}_{:05d}.png".format(outname, buffer.batch_count)
            print("- Writing to ", outpath)
            cv2.imwrite(outpath, rank_pooled)

        buffer.enqueue(frame)
    cap.release()


def run1(vidurl, outname, **kwargs):
    print(vidurl, outname)

if __name__ == '__main__':
    ### Default dir ###
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    
    vid_dir = os.path.join(proj_dir, "data/vid_src/")

    ### Arg parser ###
    parser = argparse.ArgumentParser(description="Welcome to approximated rank pooling demo.")
    parser.add_argument('-s', '--source', type=str, required=True,
        help="Videos dataset top directory. The videos should be grouped by subfolder per class/target.")
    parser.add_argument('-d', '--dest', type=str, required=True,
        help="Directory to save output. Outputs are subfolder named after each class, containing processed frames")
    parser.add_argument('--prefix', type=str, default='arprgb',
        help="Prefix naming for video output. Will auto append '_' after prefix.")
    parser.add_argument('--n_jobs', type=int, default=8)
    args = parser.parse_args()

    src = args.source
    dest = args.dest
    prefix = args.prefix
    n_jobs = args.n_jobs

    if not os.path.isdir(src):
        raise FileNotFoundError("Directory doesn't exists for "
            "source: %s" %src)

    def _outnames(vidurls, outfolder, prefix):
        fnames = []
        for url in vidurls:
            basevid, ext = os.path.splitext(os.path.basename(url))
            basevid = basevid.replace(' ','_').replace('.', '').lower()
            
            outvid = "{}_{}".format(prefix, basevid)
            outvid = os.path.join(outfolder, outvid)
            
            fnames.append(outvid)

        return fnames

    ### Main ###
    print("Found video files in:")

    for folder in os.listdir(src):  # folder per class/target
        folder_ = os.path.join(src, folder)
        if not os.path.isdir(folder_):
            continue
        vidurls = search_videos_recursively(folder_)

        print("- %s: %d files" %(folder, len(vidurls)))

        outfolder = os.path.join(dest, folder)
        try: 
            os.mkdir(outfolder)
        except (FileExistsError, OSError):
            pass  # folder has been created previously

        outpaths = _outnames(vidurls, outfolder, prefix)

        pool = Pool(n_jobs)
        args = list(zip(vidurls, outpaths))
        pool.starmap(run, args)
