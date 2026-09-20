# Quiz 3 進階測試結果

共 47 組：OK 37、WARN 1、FAIL 9

判定：角點平均誤差 < 20.0px 且相似度 ≥ 0.5 → OK；< 50.0px → WARN；其餘或找不到 → FAIL

| 檔案 | 分組 | 參數 | 偵測方法 | 角點誤差(px) | 相似度 | 結果 | 說明 |
|---|---|---|---|---:|---:|---|---|
| tilt_y_10 | tilt_y | 10 | canny/approx | 0.7 | 0.987 | OK | 角點準確，校正成功 |
| tilt_y_20 | tilt_y | 20 | canny/approx | 0.4 | 0.987 | OK | 角點準確，校正成功 |
| tilt_y_30 | tilt_y | 30 | canny/approx | 0.4 | 0.977 | OK | 角點準確，校正成功 |
| tilt_y_40 | tilt_y | 40 | canny/approx | 1.5 | 0.923 | OK | 角點準確，校正成功 |
| tilt_y_50 | tilt_y | 50 | canny/approx | 2.3 | 0.897 | OK | 角點準確，校正成功 |
| tilt_y_60 | tilt_y | 60 | canny/approx | 1.7 | 0.895 | OK | 角點準確，校正成功 |
| tilt_y_70 | tilt_y | 70 | canny/approx | 2.4 | 0.817 | OK | 角點準確，校正成功 |
| tilt_y_80 | tilt_y | 80 | canny/approx | 2.4 | 0.512 | OK | 角點準確，校正成功 |
| tilt_y_85 | tilt_y | 85 | none |  |  | FAIL | 找不到符合條件的四邊形（文件面積過小 <3%、或邊緣/二值化未形成封閉外輪廓） |
| tilt_y_88 | tilt_y | 88 | none |  |  | FAIL | 找不到符合條件的四邊形（文件面積過小 <3%、或邊緣/二值化未形成封閉外輪廓） |
| tilt_x_10 | tilt_x | 10 | canny/approx | 0.9 | 0.983 | OK | 角點準確，校正成功 |
| tilt_x_20 | tilt_x | 20 | canny/approx | 0.5 | 0.987 | OK | 角點準確，校正成功 |
| tilt_x_30 | tilt_x | 30 | canny/approx | 0.7 | 0.984 | OK | 角點準確，校正成功 |
| tilt_x_40 | tilt_x | 40 | canny/approx | 0.7 | 0.98 | OK | 角點準確，校正成功 |
| tilt_x_50 | tilt_x | 50 | canny/approx | 1.0 | 0.958 | OK | 角點準確，校正成功 |
| tilt_x_60 | tilt_x | 60 | canny/approx | 2.6 | 0.884 | OK | 角點準確，校正成功 |
| tilt_x_70 | tilt_x | 70 | canny/approx | 3.0 | 0.779 | OK | 角點準確，校正成功 |
| tilt_x_80 | tilt_x | 80 | canny/approx | 2.5 | 0.608 | OK | 角點準確，校正成功 |
| tilt_x_85 | tilt_x | 85 | canny/approx | 3.0 | 0.531 | OK | 角點準確，校正成功 |
| tilt_x_88 | tilt_x | 88 | none |  |  | FAIL | 找不到符合條件的四邊形（文件面積過小 <3%、或邊緣/二值化未形成封閉外輪廓） |
| tilt_xy_30_30 | tilt_xy | 60 | canny/approx | 0.7 | 0.97 | OK | 角點準確，校正成功 |
| tilt_xy_45_30 | tilt_xy | 75 | canny/approx | 1.2 | 0.95 | OK | 角點準確，校正成功 |
| tilt_xy_60_45 | tilt_xy | 105 | canny/approx | 1.5 | 0.886 | OK | 角點準確，校正成功 |
| roll_15_y30 | roll | 15 | canny/approx | 1.1 | 0.97 | OK | 角點準確，校正成功 |
| roll_45_y30 | roll | 45 | canny/approx | 2.1 | 0.954 | OK | 角點準確，校正成功 |
| roll_90_y30 | roll | 90 | canny/approx | 0.6 | 0.97 | OK | 角點準確，校正成功 |
| bg_wood_y30 | background | wood | otsu/approx | 3.9 | 0.929 | OK | 角點準確，校正成功 |
| bg_cluttered_y30 | background | cluttered | otsu/approx | 82.3 | 0.294 | FAIL | 偵測到錯誤的四邊形（角點嚴重偏離真實文件） |
| bg_light_y30 | background | light | canny/approx | 1.8 | 0.941 | OK | 角點準確，校正成功 |
| bg_gradient_y30 | background | gradient | canny/approx | 2.2 | 0.96 | OK | 角點準確，校正成功 |
| bg_cluttered_y50 | background | cluttered_y50 | otsu/approx | 90.1 | 0.238 | FAIL | 偵測到錯誤的四邊形（角點嚴重偏離真實文件） |
| lowcontrast_60_y30 | contrast | 0.6 | canny/approx | 0.8 | 0.981 | OK | 角點準確，校正成功 |
| lowcontrast_80_y30 | contrast | 0.8 | canny_low/approx | 0.4 | 0.972 | OK | 角點準確，校正成功 |
| lowcontrast_90_y30 | contrast | 0.9 | canny_low/approx | 1.7 | 0.941 | OK | 角點準確，校正成功 |
| lowcontrast_95_y30 | contrast | 0.95 | otsu/approx | 4.8 | 0.837 | OK | 角點準確，校正成功 |
| lowcontrast_98_y30 | contrast | 0.98 | none |  |  | FAIL | 找不到符合條件的四邊形（文件面積過小 <3%、或邊緣/二值化未形成封閉外輪廓） |
| shadow_y30 | shadow | 30 | canny/approx | 0.8 | 0.759 | OK | 角點準確，校正成功 |
| shadow_y50 | shadow | 50 | canny/approx | 1.8 | 0.795 | OK | 角點準確，校正成功 |
| noise_25_y30 | noise | 25 | canny/approx | 0.5 | 0.98 | OK | 角點準確，校正成功 |
| noise_50_y30 | noise | 50 | otsu/approx | 3.0 | 0.945 | OK | 角點準確，校正成功 |
| blur_5_y30 | blur | 5 | canny/approx | 0.9 | 0.97 | OK | 角點準確，校正成功 |
| blur_11_y30 | blur | 11 | canny/approx | 1.3 | 0.967 | OK | 角點準確，校正成功 |
| occluded_corner_y30 | occlusion | corner | canny_low/approx | 64.4 | 0.307 | FAIL | 偵測到錯誤的四邊形（角點嚴重偏離真實文件） |
| occluded_finger_y30 | occlusion | finger | canny/approx | 23.6 | 0.301 | WARN | 角點有偏移，校正結果有變形或相似度偏低 |
| cutoff_y30 | cutoff | edge | otsu/approx | 62.0 | 0.323 | FAIL | 偵測到錯誤的四邊形（角點嚴重偏離真實文件）；四邊形貼近影像邊界（文件可能被截斷） |
| far_scale_0.5_y30 | scale | 0.5 | canny/approx | 0.8 | 0.943 | OK | 角點準確，校正成功 |
| far_scale_0.25_y30 | scale | 0.25 | none |  |  | FAIL | 找不到符合條件的四邊形（文件面積過小 <3%、或邊緣/二值化未形成封閉外輪廓） |

## 失效邊界分析

- **繞垂直軸傾斜（左右側拍）**：10:OK、20:OK、30:OK、40:OK、50:OK、60:OK、70:OK、80:OK、85:FAIL、88:FAIL → 最後連續成功 80，第一個非 OK：85
- **繞水平軸傾斜（俯視/仰視）**：10:OK、20:OK、30:OK、40:OK、50:OK、60:OK、70:OK、80:OK、85:OK、88:FAIL → 最後連續成功 85，第一個非 OK：88
- **平面旋轉**：15:OK、45:OK、90:OK → 測試範圍內全部成功
- **低對比（fade 越大越接近背景）**：0.6:OK、0.8:OK、0.9:OK、0.95:OK、0.98:FAIL → 最後連續成功 0.95，第一個非 OK：0.98
- **雜訊 sigma**：25:OK、50:OK → 測試範圍內全部成功
- **模糊 kernel**：5:OK、11:OK → 測試範圍內全部成功
- **文件縮放（越小越遠）**：0.25:FAIL、0.5:OK → 最後連續成功 None，第一個非 OK：0.25
