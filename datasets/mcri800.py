import glob
import os
import re
import pandas as pd
import os.path as osp
from .bases import BaseImageDataset
from IPython import embed

class MCRI800(BaseImageDataset):
    """
    Dataset statistics:
    ----------------------------------------
    subset   | # ids | # images | # cameras
    ----------------------------------------
    train    |   801 |    80096 |         1
    query    |   287 |      824 |       583
    gallery  |   288 |    26885 |      6077
    ----------------------------------------
    """

    dataset_dir = 'MCRI800'  

    def __init__(self, root='', verbose=True, **kwargs):
        super(MCRI800, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir) # 数据集的根目录
        self.train_dir = osp.join(self.dataset_dir, 'train_all') # 测试部分
        self.query_dir = osp.join(self.dataset_dir, 'query_new') # 查询部分
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_new') # 测试集
        self.cidfile_path =osp.join(self.dataset_dir, 'pid_index_time_camid.csv') # cam_id 文件位置

        self._check_before_run()
        train = self._process_dir(self.train_dir, relabel=True,name='train')
        query = self._process_dir(self.query_dir, relabel=False,name='query')
        gallery = self._process_dir(self.gallery_dir, relabel=False,name='gallery')

        if verbose:
            print("=>MCRI800  loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)
        
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False,name=None):
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()] # 获取所有子文件夹的名称 文件夹的名称==id
        image_extensions = ['.jpg', '.png', '.jpeg']  # 图片文件的扩展名列表
        img_paths=[] # 所有图片的地址
        pid_container = set()
        for paths in sub_folders:
            pid=int(paths)
            paths=osp.join(dir_path, paths)
            img_paths.append(glob.glob(osp.join(paths, '*.jpg'))) # 获取目录下所有.jpg文件的路径列表
            # image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions] # 或者用这个获取目录下所有.jpg文件的路径列表
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}  # 交换 k v 值
        
        dataset = []
        view_container = set()
        count=0
        
        viewid=0
        if name == 'train':
            camid=0
        elif name =='gallery':
            camid=1
        elif name=='query':
            camid=2
        missingcid_count=0
        df = pd.read_csv(self.cidfile_path)
        for img_ids in img_paths:
            for img_path in img_ids:
                pid = int(img_path.split('/')[-2])
                if name != 'train':
                    img_name = re.split('/|\.', img_path)[-2]
                    id_index = int(img_name.split('_')[-1])
                    # 获取指定id和id_index的cam_id
                    try:
                        id_index_row = df[(df['id'] == pid) & (df['id_index'] == id_index)]
                        camid = id_index_row['cam_id'].values[0]
                    except Exception as e:# 写一个except
                        missingcid_count=missingcid_count+1
                        camid = -1
                        continue
                    
                assert 0 <= pid <= 1089  # pid == 0 means background
                if relabel: pid = pid2label[pid]
                view_container.add(viewid)
                dataset.append((img_path, pid, camid, viewid))
        print( 'samples without cid annotations:',missingcid_count)
        return dataset
      
if __name__=='__main__':
    dataset=MCRI800(root="/data/yuchengjin/datasets",verbose=False)
    