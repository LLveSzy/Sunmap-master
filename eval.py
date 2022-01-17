from models import create_model
from multiprocessing import Pool
from options.val_options import ValOptions


if __name__ == '__main__':
    opt = ValOptions().parse()
    model = create_model(opt)
    val_type = opt.val_type
    if val_type == 'volumes2':
        model.eval_two_volumes_maxpool()
    elif val_type == 'cubes':
        model.eval_volumes_batch()
    elif val_type == 'segment':
        imgs = model.test_3D_volume()
        pool = Pool(processes=model.opt.process)
        pool.map(model.segment_brain_batch, imgs)
        pool.close()
        pool.join()


