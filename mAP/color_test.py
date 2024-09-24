# if show_animation:
#                 # 获取图像高度和宽度
#                 height, widht = img.shape[:2]
#                 # colors (OpenCV works with BGR)
#                 white = (255,255,255)
#                 light_red = (30,30,255)
#                 light_blue = (255,200,100)
#                 # 1st line
#                 # 设置文本绘制的边距和垂直位置
#                 margin = 10
#                 v_pos = int(height - margin - (bottom_border / 2.0))
#                 # 在图像上绘制图像名称
#                 text = "Image: " + ground_truth_img[0] + " "
#                 img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
#                 # 在图像上绘制当前处理的类别索引和类别名称
#                 text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
#                 img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
#                 # 如果存在重叠（IoU），则根据检测结果获取颜色
#                 if ovmax != -1:
#                     color = get_color(class_name, "detection", status, ovmax, min_overlap)
#                     if status == "INSUFFICIENT OVERLAP":
#                         text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
#                     else:
#                         text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                
#                     img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
#                 # 2nd line
#                 v_pos += int(bottom_border / 2.0)
#                 rank_pos = str(idx+1) # rank position (idx starts at 0)
#                 text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(detection["confidence"])*100)
#                 img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
#                 color = get_color(class_name, "detection", status, ovmax, min_overlap)
#                 if status == "MATCH!":
#                     color = get_color(class_name, "ground_truth", status, ovmax, min_overlap)
#                 text = "Result: " + status + " "
#                 img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 if ovmax > 0: # if there is intersections between the bounding-boxes
#                     bbgt = [ int(round(float(x))) for x in gt_match["bbox"].split() ]
#                     color_gt = get_color(class_name, "ground_truth", status, ovmax, min_overlap)
#                     cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),color_gt,2)
#                     cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),color_gt,2)
#                     cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, color_gt, 1, cv2.LINE_AA)
#                 bb = [int(i) for i in bb]
#                 cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
#                 cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
#                 cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
#                 # # show image
#                 # cv2.imshow("Animation", img)
#                 # cv2.waitKey(20) # show for 20 ms
#                 # save image to output
#                 output_img_path = output_files_path + "/images/detections_one_by_one/" + class_name + "_detection" + str(idx) + ".jpg"
#                 cv2.imwrite(output_img_path, img)
#                 # save the image with all the objects drawn to it
#                 cv2.imwrite(img_cumulative_path, img_cumulative)