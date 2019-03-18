import logging
import os
import tempfile

import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import Keypoints


def do_detail_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger('maskrcnn_benchmark.inference')

    logger.info('Preparing results for COCO format')
    detail_results = {}
    if 'bbox' in iou_types:
        logger.info('Preparing bbox results')
        detail_results['bbox'] = prepare_for_detail_detection(predictions, dataset)
    if 'segm' in iou_types:
        logger.info('Preparing segm results')
        detail_results['segm'] = prepare_for_detail_segmentation(predictions, dataset)
    if 'keypoints'in iou_types:
        logger.info('Preparing keypoints results')
        detail_results['keypoints'] = prepare_for_detail_keypoints(predictions, dataset)

    results = DetailResults(*iou_types)
    logger.info('Evaluation predictions')
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + '.json')
                #
            res = evaluate_predictions_on_detail(
                dataset.detail, detail_results[iou_type], file_path, iou_type
            )
            results.update(res)
    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, 'detail_results.pth'))
    return results, detail_results


def prepare_for_detail_detection(predictions, dataset):
    """

    :param predictions: list(BoxList)
    :param dataset: dataset.DetailDataset
    :return: list
    """

    detail_results = []
    # prediction is a BoxList
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.idx_to_img[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        # TODO: Check dataset type and available methods
        image_width = img_info['width']
        image_height = img_info['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()

    # TODO
    return detail_results


def prepare_for_detail_segmentation(predictions, dataset):
    # TODO
    pass


def prepare_for_detail_keypoints(predictions, dataset):
    # TODO
    pass


class DetailResults(object):
    # TODO
    pass


def evaluate_predictions_on_detail(detail, param, file_path, iou_type):
    # TODO
    pass


def check_expected_results(results, expected_results, expected_results_sigma_tol):
    # TODO
    pass
