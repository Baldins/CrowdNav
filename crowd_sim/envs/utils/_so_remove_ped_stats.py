
def remove_ped_stats(data_set, run):

  if data_set == 'eth_test':
    # agent 25 is peculiar; agent 3 runs him over.
    remove_ped_frames_start = {17:0,\
                               21:17, \
                               3:0, \
                               10:0, \
                               9:0, \
                               8:0, 15:0, 23:0, 28:0}
    remove_ped_speeds = {17:6., \
                         21:8.74, \
                         3:6.74, \
                         10:7.5, \
                         9:8., \
                         8:11.4, 15:11.4, 23:11.4, 28:11.4}
  elif data_set == 'eth_train':
    # runs = {'run':(agent, start, num_steps, speed)}
    # further investigate: not 24,
    runs = {'run_0_1':(0, 0, 30, 20), \
          'run_0_2':(0, 20, 20, 15), \
          'run_0_3':(0, 40, 20, 22), \
          'run_1_full':(1, 0, 11, 15), \
          'run_1_1':(1, 0, 10, 15), \
          'run_2_full':(2, 0, 10, 18), \
          'run_2_1':(2, 0, 10, 18), \
          'run_3_full':(3, 0, 20, 18), \
          'run_3_1':(3, 0, 20, 10), \

          'run_4_full':(4, 0, 20, 18), \
          'run_4_1':(4, 0, 20, 14), \
          'run_4_2':(4, 14, 20, 18), \
          'run_4_3':(4, 30, 20, 18), \
          'run_4_4':(4, 34, 15, 18), \
          'run_5_full':(5, 0, 20, 12), \
          'run_5_1':(5, 0, 20, 8), \
          'run_5_2':(5, 20, 20, 8), \
          'run_7_full':(7, 0, 20, 12), \
          'run_7_1':(7, 0, 20, 12), \
          'run_10_full':(10, 50, 14, 21), \
          'run_10_1':(10, 50, 14, 21), \
          'run_11_full':(11, 0, 20, 21), \
          'run_11_1':(11, 0, 20, 21), \
          'run_11_2':(11, 20, 20, 21), \
          'run_11_3':(11, 30, 25, 25), \
          'run_12_full':(12, 5, 20, 20), \
          'run_12_1':(12, 5, 20, 25), \
          'run_12_2':(12, 30, 15, 18), \
          'run_12_2':(12, 30, 15, 18), \
          'run_14_full':(14, 0, 15, 22), \
          'run_14_1':(14, 0, 15, 18), \
          'run_17_full':(17, 0, 15, 15), \
          'run_17_1':(17, 0, 15, 15), \
          'run_18_full':(18, 0, 15, 18), \
          'run_18_1':(18, 0, 15, 18), \
          'run_18_2':(18, 20, 15, 18), \
          'run_20_full':(20, 30, 5, 27), \
          'run_20_1':(20, 30, 5, 27), \
          'run_20_2':(20, 35, 5, 27), \
          'run_20_3':(20, 40, 5, 27), \
          'run_20_4':(20, 45, 5, 27), \
          'run_20_5':(20, 50, 5, 27), \
          'run_21_1':(21, 60, 15, 12), \
          'run_22_1':(22, 0, 22, 8), \

          'run_24_full':(24, 0, 15, 10), \
          'run_24_1':(24, 0, 20, 13), \
          'run_24_2':(24, 20, 15, 20), \
          'run_25_full':(25, 0, 7, 12), \
          'run_25_1':(25, 0, 12, 12), \
          'run_25_2':(25, 35, 13, 22), \
          'run_31_1':(31, 58, 10, 18), \

          'run_32_full':(32, 0, 20, 10), \
          'run_32_1':(32, 0, 20, 12), \
          'run_32_2':(32, 10, 20, 16), \
          'run_32_3':(32, 30, 15, 18), \
          'run_32_4':(32, 40, 15, 22), \
          'run_34_full':(34, 0, 15, 25), \
          'run_34_1':(34, 0, 15, 25), \

          'run_40_full':(40, 0, 20, 22), \
          'run_40_1':(40, 3, 25, 22), \
          'run_48_full':(48, 15, 9, 25), \
          'run_48_1':(48, 15, 9, 25), \

          'run_49_full':(49, 0, 50, 20), \
          'run_49_1':(49, 0, 50, 12), \
          'run_49_2':(49, 50, 20, 12), \
          'run_50_1':(50, 40, 10, 30), \
          'run_50_2':(50, 50, 10, 30), \
          'run_52_full':(52, 0, 25, 12), \
          'run_52_1':(52, 0, 30, 12), \
          'run_52_2':(52, 30, 25, 19), \

          'run_53_full':(53, 34, 8, 35), \
          'run_53_1':(53, 34, 10, 32), \
          'run_53_2':(53, 44, 10, 32), \
          'run_54_full':(54, 0, 10, 12), \
          'run_54_1':(54, 0, 18, 12), \
          'run_55_1':(55, 50, 25, 18), \
          'run_55_2':(55, 75, 20, 24), \

          'run_57_full':(57, 0, 15, 20), \
          'run_57_1':(57, 0, 25, 20), \
          'run_57_2':(57, 25, 18, 18), \
          'run_57_3':(57, 25, 18, 18), \
          'run_57_4':(57, 45, 15, 20), \
          'run_58_full':(58, 0, 15, 20), \
          'run_58_1':(58, 0, 25, 20), \
          'run_59_1':(59, 0, 20, 5), \
          'run_59_2':(59, 20, 20, 20), \
          'run_63_full':(63, 0, 8, 34), \
          'run_63_1':(63, 0, 15, 25), \
          'run_64_1':(64, 0, 15, 20), \
          'run_64_2':(64, 15, 15, 25), \
          'run_64_3':(64, 25, 17, 20), \
          'run_66_1':(66, 0, 8, 30), \
          'run_66_2':(66, 8, 8, 34), \
          'run_67_1':(67, 60, 25, 24), \
          'run_69_full':(69, 0, 15, 10), \
          'run_69_1':(69, 0, 15, 10), \
          'run_71_full':(71, 0, 15, 20), \
          'run_71_1':(71, 0, 15, 20), \
          'run_71_2':(71, 15, 15, 20), \
          'run_71_3':(71, 30, 15, 25), \

          'run_73_full':(73, 0, 7, 20), \
          'run_73_1':(73, 0, 10, 30), \
          'run_73_2':(73, 10, 6, 30), \
          'run_74_full':(74, 0, 15, 20), \
          'run_74_1':(74, 0, 20, 20), \
          'run_74_2':(74, 10, 22, 20), \

          'run_75_full':(75, 30, 15, 22), \
          'run_75_1':(75, 30, 15, 20), \
          'run_75_2':(75, 45, 12, 35), \
          'run_75_3':(75, 57, 12, 30), \
          'run_75_4':(75, 60, 15, 20), \
          'run_75_5':(75, 75, 20, 30), \
          'run_80_full':(80, 0, 10, 30), \
          'run_80_1':(80, 0, 10, 30), \
          'run_80_2':(80, 10, 9, 30), \
          'run_82_full':(82, 0, 15, 30), \
          'run_82_1':(82, 0, 15, 20), \
          'run_84_full':(84, 0, 15, 35), \
          'run_84_1':(84, 3, 17, 15), \
          'run_84_2':(84, 20, 17, 25), \
          'run_84_3':(84, 37, 15, 25), \
          'run_84_4':(84, 50, 20, 25), \

          'run_85_full':(85, 0, 20, 5), \
          'run_85_1':(85, 0, 30, 5), \
          'run_87_full':(87, 30, 7, 40), \
          'run_87_1':(87, 30, 10, 40), \
          'run_87_2':(87, 40, 8, 10), \
          'run_87_3':(87, 50, 10, 40), \
          'run_88_full':(88, 0, 10, 25), \
          'run_88_1':(88, 0, 20, 25), \
          'run_88_2':(88, 18, 20, 25), \
          'run_88_3':(88, 30, 20, 20), \
          'run_88_4':(88, 40, 18, 25), \
          'run_88_5':(88, 70, 20, 25), \
          'run_91_full':(91, 0, 20, 30), \
          'run_91_1':(91, 0, 22, 20), \
          'run_91_2':(91, 22, 7, 30), \
          'run_91_3':(91, 30, 7, 30), \
          'run_92_1':(92, 0, 21, 10), \
          'run_94_1':(94, 0, 12, 25), \

          'run_98_full':(98, 0, 25, 5), \
          'run_98_1':(98, 0, 25, 10), \
          'run_98_2':(98, 30, 30, 30), \
          'run_98_3':(98, 60, 10, 30), \
          'run_99_full':(99, 45, 10, 30), \
          'run_99_4':(99, 45, 15, 30), \
          'run_99_5':(99, 60, 10, 30), \
          'run_100_full':(100, 0, 20, 30), \
          'run_100_1':(100, 0, 20, 20), \
          'run_100_2':(100, 20, 10, 20), \
          'run_101_1':(101, 0, 25, 25), \
          'run_101_2':(101, 25, 10, 20), \
          'run_102_full':(102, 0, 6, 30), \
          'run_102_1':(102, 0, 10, 35), \
          'run_102_2':(102, 10, 10, 30), \
          'run_102_3':(102, 20, 10, 30), \
          'run_102_4':(102, 30, 20, 30), \
          'run_102_5':(102, 40, 20, 30), \
          'run_104_1':(104, 0, 15, 10), \
          'run_105_full':(105, 00, 7, 40), \
          'run_105_1':(105, 0, 7, 30), \
          'run_105_3':(105, 25, 14, 40), \
          'run_105_4':(105, 40, 14, 40), \
          'run_106_full':(106, 0, 8, 40), \
          'run_106_1':(106, 0, 10, 30), \
          'run_106_2':(106, 10, 8, 30), \
          'run_107_full':(107, 0, 18, 30), \
          'run_107_1':(107, 10, 18, 20), \
          'run_107_2':(107, 28, 18, 20), \
          'run_107_3':(107, 46, 18, 20), \
          'run_107_4':(107, 64, 30, 20), \
          'run_108_full':(108, 20, 20, 20), \
          'run_108_1':(108, 0, 20, 10), \
          'run_108_2':(108, 20, 15, 25), \
          'run_109_full':(109, 0, 5, 18), \
          'run_109_1':(109, 0, 15, 20), \
          'run_109_2':(109, 10, 15, 18), \
          'run_109_3':(109, 20, 15, 21), \
          'run_110_1':(110, 0, 20, 12), \
          'run_110_2':(110, 20, 15, 18), \
          'run_110_3':(110, 40, 15, 18), \
          'run_111_1':(111, 0, 15, 25), \
          'run_111_2':(111, 15, 15, 22), \
          'run_112_full':(112, 40, 10, 22), \
          'run_112_3':(112, 40, 20, 22), \
          'run_112_4':(112, 60, 10, 22), \
          'run_114_full':(114, 0, 7, 35), \
          'run_114_1':(114, 0, 10, 25), \
          'run_115_full':(115, 0, 25, 35), \
          'run_115_1':(115, 0, 25, 25), \
          'run_115_2':(115, 25, 15, 25), \
          'run_116_full':(116, 0, 40, 15), \
          'run_116_1':(116, 0, 40, 15), \
          'run_117_full':(117, 60, 5, 40), \
          'run_117_1':(117, 60, 10, 35), \
          'run_117_2':(117, 70, 10, 35), \
          'run_118_full':(118, 0, 22, 40), \
          'run_118_1':(118, 0, 50, 10), \
          'run_118_2':(118, 50, 10, 40), \
          'run_118_3':(118, 65, 10, 30), \
          'run_119_full':(119, 0, 5, 45), \
          'run_119_1':(119, 0, 6, 40), \
          'run_119_2':(119, 6, 6, 45), \
          'run_120_full':(120, 0, 8, 45), \
          'run_120_2':(120, 10, 10, 45), \
          'run_120_3':(120, 20, 10, 45), \
          'run_121_1':(121, 45, 15, 27), \
          'run_125_full':(125, 0, 8, 35), \
          'run_125_1':(125, 0, 12, 35), \
          'run_125_2':(125, 12, 12, 35), \
          'run_126_full':(126, 0, 50, 40), \
          'run_126_1':(126, 0, 50, 20), \
          'run_126_2':(126, 50, 12, 35), \
          'run_126_3':(126, 60, 10, 40), \
          'run_127_1':(127, 0, 19, 25), \
          'run_128_1':(128, 0, 9, 40), \
          'run_129_full':(129, 0, 52, 40), \
          'run_129_1':(129, 0, 52, 20), \
          'run_129_2':(129, 52, 12, 40), \
          'run_129_3':(129, 64, 12, 40), \
          'run_130_full':(130, 13, 8, 35), \
          'run_130_1':(130, 13, 13, 30), \
          'run_130_2':(130, 26, 13, 30), \
          'run_130_3':(130, 39, 13, 30), \
          'run_131_1':(131, 52, 20, 10), \
          'run_131_2':(131, 72, 15, 20), \

          'run_135_full':(135, 0, 12, 10), \
          'run_135_1':(135, 0, 20, 15), \
          'run_138_full':(138, 0, 5, 45), \
          'run_138_1':(138, 18, 10, 20), \
          'run_138_2':(138, 48, 15, 20), \
          'run_139_1':(139, 15, 15, 20), \
          'run_139_2':(139, 37, 7, 35), \
          'run_140_1':(140, 60, 11, 45), \
          'run_141_1':(141, 60, 20, 20), \
          'run_143_full':(143, 0, 10, 30), \
          'run_145_1':(145, 0, 19, 10), \
          'run_147_full':(147, 0, 10, 10), \
          'run_147_1':(147, 0, 15, 25), \
          'run_147_2':(147, 20, 15, 20), \
          'run_147_3':(147, 35, 12, 15), \
          'run_147_4':(147, 50, 30, 15), \
          'run_148_1':(148, 0, 35, 10), \
          'run_149_1':(149, 0, 17, 20)}
    remove_ped, remove_ped_start, goal_dex, max_vel_robot  = runs[run]
#     remove_ped_frames_start = {
# 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, \
# 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 25:0, 26:0, 27:0, 28:0, 29:0, \
# 30:0, 31:0, 33:0, 34:0, 35:0, 36:0, 37:0, 38:0, 39:0, 40:0, 41:0, 42:0, \
# 43:0, 44:0, 45:0, 46:0, 47:0, 48:0, 49:0, 50:0, 51:0, 52:0, 53:0, 54:0,
# 55:0, \
# 56:0, 57:0, 58:0, 59:0, 60:0, 61:0, 62:0, 63:0, 64:0, 65:0, 66:0, 67:0, \
# 68:0, \
# 69:0, 70:0, 71:0, 72:0, 73:0, 74:0, 75:0, 76:0, 77:0, 78:0, 79:0, 80:0, \
# 81:0, \
# 82:0, 83:0, 84:0, 85:0, 86:0, 87:0, 88:0, 89:0, 90:0, 91:0, 92:0, 93:0,
# 94:0, \
# 95:0, 96:0, 97:0, 98:0, 99:0, 100:0, 101:0, 102:0, 103:0, 104:0, 105:0,
# 106:0, \
# 107:0, 108:0, 111:0, 112:0, 113:0, 114:0, 115:0, 116:0, 117:0, \
# 118:0, 119:0, 120:0, 121:0, 122:0, 123:0, 124:0, 125:0, 126:0, 127:0,
# 128:0, \
# 129:0, 130:0, 131:0, 132:0, 133:0, 134:0, 135:0, 136:0, 137:0, 138:0,
# 139:0, \
# 140:0, 141:0, 142:0, 143:0, 144:0, 145:0, 146:0, 147:0, 148:0, 149:0}
#     remove_ped_speeds = {5:12, 6:12, 7:12, \
# 8:12, 9:12, 10:12, 11:12, \
# 12:12, 13:12, 14:12, 15:12, 16:12, 17:12, 18:12, 19:12, 20:12, 21:12, 22:12, \
# 23:12, 25:12, 26:12, 27:12, 28:12, 29:12, 30:12, 31:12, 33:12, 34:12, 35:12, \
# 36:12, 37:12, 38:12, 39:12, 40:12, 41:12, 42:12, 43:12, 44:12, 45:12, 46:12, \
# 47:12, 48:12, 49:12, 50:12, 51:12, 52:12, 53:12, 54:12, 55:12, 56:12, 57:12, \
# 58:12, 59:12, 60:12, 61:12, 62:12, 63:12, 64:12, 65:12, 66:12, 67:12, 68:12, \
# 69:12, 70:12, 71:12, 72:12, 73:12, 74:12, 75:12, 76:12, 77:12, 78:12, 79:12, \
# 80:12, 81:12, 82:12, 83:12, 84:12, 85:12, 86:12, 87:12, 88:12, 89:12, 90:12, \
# 91:12, 92:12, 93:12, 94:12, 95:12, 96:12, 97:12, 98:12, 99:12, 100:12,
# 101:12, \
# 102:12, 103:12, 104:12, 105:12, 106:12, 107:12, 108:12, 111:12, 112:12, \
# 113:12, 114:12, 115:12, 116:12, 117:12, 118:12, 119:12, 120:12, 121:12, \
# 122:12, 123:12, 124:12, 125:12, 126:12, 127:12, 128:12, 129:12, 130:12, \
# 131:12, 132:12, 133:12, 134:12, 135:12, 136:12, 137:12, 138:12, 139:12, \
# 140:12, 141:12, 142:12, 143:12, 144:12, 145:12, 146:12, 147:12, 148:12,
# 149:12}
  # remove_ped_start = remove_ped_frames_start[remove_ped]
  # max_vel_robot =  remove_ped_speeds[remove_ped]

  max_vel_ped = 6.74

  return remove_ped, remove_ped_start, goal_dex, max_vel_robot, max_vel_ped






