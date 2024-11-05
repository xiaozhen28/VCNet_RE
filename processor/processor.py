import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

from IPython import embed




def do_stage1(cfg,
             model,
             center_criterion,
             train_loader,
             query_loader,
             gallery_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, 
             output_dir
             ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = 120

    logger = logging.getLogger("transreid.train_stage1")
    logger.info('start training')
    model.cuda()
    
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        model = torch.nn.parallel.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,query_methd=cfg.TEST.EVAL_METHD, muti_query_num=cfg.TEST.EVAL_NUM,simularity_threshold=cfg.TEST.SIMULARITY_TH)
    scaler = amp.GradScaler()



    model.stage=1
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view) # 模型得分，最终特征
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(output_dir,cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(output_dir,cfg.MODEL.NAME + 'stage1_{}.pth'.format(epoch)))

        # 测试阶段
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(query_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, gvid = model(img, cam_label=camids, view_label=target_view,isquery=True)
                            evaluator.update((feat, gvid, camid))

                    cmc, mAP, mCSP, mINP,_, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results By {} muti_num={}".format(cfg.TEST.EVAL_METHD,cfg.TEST.EVAL_NUM))
                    logger.info("mAP: {:.1%}".format(mAP))
                    logger.info("mCSP: {:.1%}".format(mCSP))
                    logger.info("mINP: {:.1%}".format(mINP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(query_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, vid, cam_label=camids, view_label=target_view)
                        evaluator.update((feat,vid, camid))
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(gallery_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img,vid,cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP,mCSP,mINP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results By {} muti_num={}".format(cfg.TEST.EVAL_METHD,cfg.TEST.EVAL_NUM))
                logger.info("mAP: {:.1%}".format(mAP))
                logger.info("mCSP: {:.1%}".format(mCSP))
                logger.info("mINP: {:.1%}".format(mINP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_view_stage(cfg,
             model,
             center_criterion,
             train_loader,
             query_loader,
             gallery_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, 
             output_dir
             ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = 120

    logger = logging.getLogger("transreid.train_stage1")
    logger.info('start training')
    model.cuda()
    
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        model = torch.nn.parallel.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,query_methd=cfg.TEST.EVAL_METHD, muti_query_num=cfg.TEST.EVAL_NUM,simularity_threshold=cfg.TEST.SIMULARITY_TH)
    scaler = amp.GradScaler()


    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view) # 模型得分，最终特征
                # loss = loss_fn(score, feat, target, target_cam)
                loss = loss_fn(score, feat, target_view)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target_view).float().mean()
            else:
                acc = (score.max(1)[1] == target_view).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(output_dir,cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(output_dir,cfg.MODEL.NAME + 'stage1_{}.pth'.format(epoch)))

        # 测试阶段
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(query_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, gvid = model(img, cam_label=camids, view_label=target_view,isquery=True)
                            evaluator.update((feat, gvid, camid))

                    cmc, mAP, mCSP, mINP,_, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results By {} muti_num={}".format(cfg.TEST.EVAL_METHD,cfg.TEST.EVAL_NUM))
                    logger.info("mAP: {:.1%}".format(mAP))
                    logger.info("mCSP: {:.1%}".format(mCSP))
                    logger.info("mINP: {:.1%}".format(mINP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(query_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view
                        feat = model(img, vid, cam_label=camids, view_label=target_view)
                        evaluator.update((feat,target_view, camid))
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(gallery_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view
                        feat = model(img,vid,cam_label=camids, view_label=target_view)
                        evaluator.update((feat, target_view, camid))
                cmc, mAP,mCSP,mINP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results By {} muti_num={}".format(cfg.TEST.EVAL_METHD,cfg.TEST.EVAL_NUM))
                logger.info("mAP: {:.1%}".format(mAP))
                logger.info("mCSP: {:.1%}".format(mCSP))
                logger.info("mINP: {:.1%}".format(mINP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_stage2(cfg,
             model_view,
             model,
             center_criterion,
             train_loader,
             query_loader,
             gallery_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, 
             output_dir):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = 120

    logger = logging.getLogger("transreid.train_stage2")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    model.cuda()
    model_view.cuda()
    # if torch.cuda.device_count() > 1:
    #     print('Using {} GPUs for training'.format(torch.cuda.device_count()))
    #     model = torch.nn.parallel.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,query_methd=cfg.TEST.EVAL_METHD, muti_query_num=cfg.TEST.EVAL_NUM,simularity_threshold=cfg.TEST.SIMULARITY_TH)
    scaler = amp.GradScaler()

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        model_view.eval()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output_view=model_view(img) 
                score,feat = model(img,output_view) # 模型得分，最终特征
                # loss = 0
                # for i in range(0, len(output), 4):
                #         loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=output[i+2], target_cam=output[i+3])
                # loss = loss + loss_tmp

                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
           
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, cfg.MODEL.NAME + '_stage2_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),os.path.join(output_dir, cfg.MODEL.NAME + '_stage2_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(query_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat,gvid = model(img, cam_label=camids, view_label=target_view,isquery=True)
                            evaluator.update((feat, gvid, camid))
                    cmc, mAP, mCSP, mINP,_, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results By {} muti_num={}".format(cfg.TEST.EVAL_METHD,cfg.TEST.EVAL_NUM))
                    logger.info("mAP: {:.1%}".format(mAP))
                    logger.info("mCSP: {:.1%}".format(mCSP))
                    logger.info("mINP: {:.1%}".format(mINP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(query_loader):
                    
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        output_view=model_view(img)
                        feat = model(img,output_view)
                        evaluator.update((feat,output_view, vid, camid))


                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(gallery_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        output_view=model_view(img)
                        feat = model(img,output_view)
                        evaluator.update((feat,output_view,vid, camid))
                
                cmc, mAP,mCSP,mINP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results By {} muti_num={}".format(cfg.TEST.EVAL_METHD,cfg.TEST.EVAL_NUM))
                logger.info("mAP: {:.1%}".format(mAP))
                logger.info("mCSP: {:.1%}".format(mCSP))
                logger.info("mINP: {:.1%}".format(mINP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
    


def do_inference(cfg,
                 model,
                 query_loader,
                 gallery_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,query_methd=cfg.TEST.EVAL_METHD, muti_query_num=cfg.TEST.EVAL_NUM,simularity_threshold=cfg.TEST.SIMULARITY_TH)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    with torch.no_grad():
        for n_iter, (img, vid, camid, camids, target_view, imgpath) in enumerate(query_loader):
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat,gvid = model(img,vid,cam_label=camids, view_label=target_view)
            evaluator.update((feat, gvid, camid))
            img_path_list.extend(imgpath)
        for n_iter, (img, vid, camid, camids, target_view, imgpath) in enumerate(gallery_loader):
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img,vid,cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))
            img_path_list.extend(imgpath)
    cmc, mAP,mCSP,mINP,_, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results By {} muti_num={}".format(cfg.TEST.EVAL_METHD,cfg.TEST.EVAL_NUM))
    logger.info("mAP: {:.1%}".format(mAP))
    logger.info("mCSP: {:.1%}".format(mCSP))
    logger.info("mINP: {:.1%}".format(mINP))
   
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
