# 由sql生成
labelname2cid_dct = {
    'l1': {'yphl': 0, 'bad_l2': 1},
    'l2': {'优': 0, '普': 1, '封面昏暗或模糊': 2, '画质不佳': 3, '遮挡LOGO': 4, '严重卡顿': 5},
    'l2_combine': {'优': 0, '普': 1, '封面昏暗或模糊': 2, '画质不佳': 3, '遮挡LOGO': 4, '严重卡顿': 5},
    'l2_combine2': {'优': 0, '普': 0, '封面昏暗或模糊': 0, '画质不佳': 0, '遮挡LOGO': 1, '严重卡顿': 0},
    'l2_combine3': {'优': 0, '普': 0, '封面昏暗或模糊': 1, '画质不佳': 1, '遮挡LOGO': 0, '严重卡顿': 0},
    'l2_combine4': {'优': 0, '普': 0, '封面昏暗或模糊': 0, '画质不佳': 0, '遮挡LOGO': 0, '严重卡顿': 1},
    'op_l1':{'0_others': 0, '1_喜剧&搞笑': 1, '101_潮流玩法': 2, '106_帅哥': 3, '108_美食': 4, '112_猎奇': 5,
                   '120_宠物&动物': 6, '124_新闻': 7, '129_才艺秀': 8, '140_机动车': 9, '145_评测': 10, '148_生活技巧': 11,
                   '152_游戏': 12, '155_动漫': 13, '157_宗教': 14, '166_自然': 15, '168_旅游': 16, '170_情感关系': 17,
                   '174_生活记录': 18, '180_LGBTQ': 19, '202_教育': 20, '207_数码产品': 21, '21_美女': 22,
                   '316_颜值高': 23, '32_鸡汤类': 24, '329_儿童': 25, '352_音乐': 26, '353_影视': 27, '41_舞蹈': 28,
                   '54_时尚': 29, '67_运动': 30, '88_娱乐': 31},
    'op_l1_co':{'0_others': 0, '1_喜剧&搞笑': 1, '101_潮流玩法': 2, '106_帅哥': 3, '108_美食': 4, '112_猎奇': 5,
                   '120_宠物&动物': 6, '124_新闻': 7, '129_才艺秀': 8, '140_机动车': 9, '145_评测': 10, '148_生活技巧': 11,
                   '152_游戏': 12, '155_动漫': 13, '157_宗教': 14, '166_自然': 15, '168_旅游': 16, '170_情感关系': 17,
                   '174_生活记录': 18, '180_LGBTQ': 0, '202_教育': 20, '207_数码产品': 21, '21_美女': 3,
                   '316_颜值高': 3, '32_鸡汤类': 24, '329_儿童': 25, '352_音乐': 26, '353_影视': 27, '41_舞蹈': 28,
                   '54_时尚': 23, '67_运动': 22, '88_娱乐': 19},
    'op_l2':{'0_others_0_others': 0, '1_喜剧&搞笑_10_街头实验': 1, '1_喜剧&搞笑_11_游戏片段': 2,
                    '1_喜剧&搞笑_12_喜剧表演': 3, '1_喜剧&搞笑_13_脱口秀喜剧n0710': 4, '1_喜剧&搞笑_14_搞笑影视片段': 5,
                    '1_喜剧&搞笑_15_对口型搞笑': 6, '1_喜剧&搞笑_18_Funnytext': 7, '1_喜剧&搞笑_2_悲剧沙雕': 8,
                    '1_喜剧&搞笑_3_恶搞n0710': 9, '1_喜剧&搞笑_4_吐槽': 10, '1_喜剧&搞笑_5_奇葩': 11, '1_喜剧&搞笑_6_动物': 12,
                    '1_喜剧&搞笑_7_搞笑宠物': 13, '1_喜剧&搞笑_8_对话访谈': 14, '1_喜剧&搞笑_9_搞笑宝宝': 15,
                    '101_潮流玩法_102_对口型': 16, '101_潮流玩法_103_运镜': 17, '101_潮流玩法_104_创意特效': 18,
                    '101_潮流玩法_105_挑战': 19, '101_潮流玩法_190_摄影': 20, '101_潮流玩法_216_动态壁纸': 21,
                    '101_潮流玩法_365_魔表': 22, '101_潮流玩法_366_小柿饼mv': 23, '101_潮流玩法_367_duet': 24,
                    '106_帅哥_107_帅哥': 25, '108_美食_109_食物': 26, '108_美食_110_烹饪教学': 27, '108_美食_111_探店': 28,
                    '108_美食_346_吃播': 29, '112_猎奇_115_幽灵鬼怪': 30, '112_猎奇_116_自然奇闻': 31,
                    '112_猎奇_117_历史逸闻': 32, '112_猎奇_118_未解之谜': 33, '112_猎奇_119_奇人异事': 34,
                    '112_猎奇_198_特技': 35, '112_猎奇_218_新奇实验': 36, '120_宠物&动物_121_猫': 37,
                    '120_宠物&动物_122_狗': 38, '120_宠物&动物_123_宠物其他': 39, '120_宠物&动物_200_蛇': 40,
                    '120_宠物&动物_325_鸟': 41, '120_宠物&动物_326_鱼': 42, '120_宠物&动物_327_蛇类': 43,
                    '120_宠物&动物_328_其他动物': 44, '120_宠物&动物_351_猴子': 45, '120_宠物&动物_364_牛': 46,
                    '124_新闻_125_本地新闻': 47, '124_新闻_126_环球新闻': 48, '124_新闻_354_社会新闻': 49,
                    '124_新闻_355_新冠疫情': 50, '124_新闻_356_国际时政': 51, '124_新闻_357_经济新闻': 52,
                    '124_新闻_358_法律新闻': 53, '124_新闻_359_体育新闻': 54, '129_才艺秀_130_唱歌': 55,
                    '129_才艺秀_131_魔术': 56, '129_才艺秀_132_绘画': 57, '129_才艺秀_133_乐器': 58, '129_才艺秀_134_吉他': 59,
                    '129_才艺秀_135_空翻表演': 60, '129_才艺秀_136_杂技': 61, '129_才艺秀_137_艺术': 62,
                    '129_才艺秀_138_饶舌': 63, '129_才艺秀_139_书法': 64, '129_才艺秀_337_其他才艺秀': 65,
                    '140_机动车_141_卡车': 66, '140_机动车_142_改装车': 67, '140_机动车_143_摩托': 68,
                    '140_机动车_144_豪车': 69, '140_机动车_361_汽车': 70, '145_评测_147_其他开箱': 71,
                    '148_生活技巧_149_手工定制': 72, '148_生活技巧_150_生活小技巧': 73, '148_生活技巧_151_解压': 74,
                    '148_生活技巧_194_健康小技巧': 75, '148_生活技巧_347_拍照摄影': 76, '148_生活技巧_348_视频剪辑': 77,
                    '152_游戏_153_FreeFire': 78, '152_游戏_201_吃鸡': 79, '152_游戏_212_其他游戏': 80,
                    '152_游戏_341_moba手游': 81, '155_动漫_156_动漫': 82, '155_动漫_205_卡通': 83,
                    '155_动漫_206_视频特效': 84, '157_宗教_158_印度教': 85, '157_宗教_159_伊斯兰教': 86,
                    '157_宗教_160_基督教': 87, '157_宗教_161_锡克教': 88, '157_宗教_162_佛教': 89, '157_宗教_163_耆那教': 90,
                    '157_宗教_165_宗教其他': 91, '166_自然_167_自然': 92, '168_旅游_169_旅游': 93, '168_旅游_334_自然风光': 94,
                    '168_旅游_335_城市街景': 95, '168_旅游_336_旅行见闻': 96, '170_情感关系_171_情侣/夫妻关系': 97,
                    '170_情感关系_172_情侣日常': 98, '170_情感关系_173_友谊': 99, '170_情感关系_191_家庭.': 100,
                    '170_情感关系_362_爱国': 101, '174_生活记录_175_日常生活': 102, '174_生活记录_176_工作生活记录': 103,
                    '174_生活记录_177_女性自拍': 104, '174_生活记录_178_男性自拍': 105, '174_生活记录_179_乡村生活': 106,
                    '174_生活记录_196_婚礼': 107, '174_生活记录_342_军旅生活': 108, '174_生活记录_343_自拍': 109,
                    '174_生活记录_344_聊天截屏': 110, '174_生活记录_360_多人自拍': 111, '180_LGBTQ_181_TheLGBTQ': 112,
                    '202_教育_203_英语教学': 113, '202_教育_204_通识常识': 114, '202_教育_350_演讲（宗教）': 115,
                    '207_数码产品_208_手机': 116, '207_数码产品_209_智能手表': 117, '207_数码产品_210_数码相机': 118,
                    '207_数码产品_211_电脑': 119, '207_数码产品_339_开箱': 120, '207_数码产品_340_其他产品': 121,
                    '21_美女_22_性感': 122, '21_美女_23_比基尼': 123, '21_美女_25_高颜值': 124, '21_美女_26_大胸': 125,
                    '21_美女_28_模特': 126, '21_美女_29_街拍': 127, '21_美女_30_跳舞': 128, '21_美女_31_女明星': 129,
                    '316_颜值高_317_高颜值女生': 130, '316_颜值高_318_帅气小哥': 131, '32_鸡汤类_214_问好': 132,
                    '32_鸡汤类_33_爱情鸡汤': 133, '32_鸡汤类_33_情感鸡汤': 134, '32_鸡汤类_34_伤感鸡汤': 135,
                    '32_鸡汤类_345_鸡汤': 136, '32_鸡汤类_35_励志鸡汤': 137, '32_鸡汤类_36_态度鸡汤': 138,
                    '32_鸡汤类_40_音乐歌词类': 139, '329_儿童_330_萌娃': 140, '329_儿童_331_育儿知识': 141,
                    '352_音乐_324_电音': 142, '352_音乐_89_音乐/演唱会现场': 143, '352_音乐_90_音乐MV': 144,
                    '352_音乐_94_音乐推荐': 145, '352_音乐_99_对口型唱歌': 146, '353_影视_91_综艺秀': 147,
                    '353_影视_93_影视片段': 148, '41_舞蹈_319_蹦迪': 149, '41_舞蹈_44_街舞': 150, '41_舞蹈_47_肚皮舞': 151,
                    '41_舞蹈_48_古典舞': 152, '41_舞蹈_49_宝莱坞舞蹈': 153, '41_舞蹈_50_搞笑舞蹈': 154, '41_舞蹈_51_手指舞': 155,
                    '41_舞蹈_52_乡村舞蹈': 156, '41_舞蹈_53_舞蹈其他': 157, '54_时尚_193_化妆品评测': 158,
                    '54_时尚_195_曼海蒂': 159, '54_时尚_320_泳装比基尼': 160, '54_时尚_321_时尚其他': 161,
                    '54_时尚_363_珠宝': 162, '54_时尚_55_纹身': 163, '54_时尚_56_美甲': 164, '54_时尚_57_发型': 165,
                    '54_时尚_58_美妆': 166, '54_时尚_59_搭配': 167, '54_时尚_64_微整形': 168, '54_时尚_66_特殊化妆': 169,
                    '67_运动_192_瑜伽': 170, '67_运动_197_Kabaddi': 171, '67_运动_322_潜水': 172, '67_运动_63_美体': 173,
                    '67_运动_69_健美': 174, '67_运动_70_篮球': 175, '67_运动_71_足球': 176, '67_运动_72_板球': 177,
                    '67_运动_73_羽毛球': 178, '67_运动_76_球类运动其他': 179, '67_运动_77_排球': 180, '67_运动_78_水上运动': 181,
                    '67_运动_79_户外运动': 182, '67_运动_80_跑酷': 183, '67_运动_81_极限运动': 184, '67_运动_82_田径': 185,
                    '67_运动_86_摔跤': 186, '67_运动_87_体育其他': 187, '88_娱乐_100_表演情景剧': 188, '88_娱乐_323_韩流': 189,
                    '88_娱乐_324_电音': 190, '88_娱乐_89_音乐/演唱会现场': 191, '88_娱乐_90_音乐MV': 192,
                    '88_娱乐_91_综艺秀': 193, '88_娱乐_92_娱乐八卦': 194, '88_娱乐_93_影视片段': 195, '88_娱乐_96_明星写真': 196,
                    '88_娱乐_97_韩国男团': 197, '88_娱乐_98_韩国女团': 198, '88_娱乐_99_对口型唱歌': 199}
}
