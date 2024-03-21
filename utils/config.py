# This file is used to configure the training parameters for each task
class Config_SAMUS:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../../dataset/SAMUS/"  
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = "./checkpoints/XXXXX.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 12                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-EchocardiographyLV-CAMUS"               # the file name of training set
    val_split = "val-EchocardiographyLV-CAMUS"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CT:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../../dataset/CT/"  
    save_path = "./checkpoints/Compare/"
    result_path = "./result/Compare/"
    tensorboard_path = "./tensorboard/Compare/"
    load_path = "./checkpoints/XXXXX.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                      # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train_Compare"          # the file name of training set
    val_split = "val_Compare"              # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 1                       # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CT5M:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../../dataset/CT5M/"
    save_path = "./checkpoints/CT5M/"
    result_path = "./result/CT5M/"
    tensorboard_path = "./tensorboard/CT5M/"
    load_path = "./checkpoints/XXXXX.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 50                         # number of total epochs to run (default: 400)
    batch_size = 12                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "all_train"    # the file name of training set
    val_split = "all_val"        # the file name of testing set
    test_split = "test"      # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 5000                    # the frequency of evaluate the model
    save_freq = 5000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# ----------------------------------------------------------------------------------------------------------------
class Config_COVID19:
    data_path = "../../dataset/CT/"
    data_subpath = "../../dataset/CT/Covid-19-20/"
    save_path = "./checkpoints/CT/COVID-19CTscans/"
    result_path = "./result/CT/COVID-19CTscans/"
    tensorboard_path = "./tensorboard/CT/COVID-19CTscans/"
    load_path = "./xxxx"
    visual_result_path = "./result/CT/COVID-19CTscans/"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 200                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256               # the input size of model
    train_split = "train_COVID19C1"        # the file name of training set
    val_split = "val_COVID19C1"
    test_split = "all_test_COVID-19CTscans_C1"           # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000             # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"     # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "TransFuse"

class Config_WORD:
    data_path = "../../dataset/CT/"
    save_path = "./checkpoints/CT/WORD/"
    result_path = "./result/CT/WORD/"
    tensorboard_path = "./tensorboard/CT/WORD/"
    load_path = "./xxxx"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 300                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "all_train_WORD_C4"        # the file name of training set
    val_split = "all_val_WORD_C4"
    test_split = "all_test_WORD_C5"           # the file name of testing set
    crop = None            # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_patient"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAMCT"

class Config_FUMPE:
    data_path = "../../dataset/CT/"
    data_subpath = "../../dataset/CT/FUMPE/"
    save_path = "./checkpoints/CT/FUMPE/"
    result_path = "./result/CT/FUMPE/"
    tensorboard_path = "./tensorboard/CT/FUMPE/"
    load_path = "./xxxx"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 200                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train_FUMPE"        # the file name of training set
    val_split = "val_FUMPE"
    test_split = "test_FUMPE"           # the file name of testing set
    crop = None              # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_LCTSC:
    data_path = "../../dataset/CT/"
    dats_subpath = "../../dataset/CT/LCTSC/"
    save_path = "./checkpoints/CT/LCTSC/"
    result_path = "./result/CT/LCTSC/"
    tensorboard_path = "./tensorboard/CT/LCTSC/"
    load_path = "./xxxx"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 200                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 5                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train_LCTSC"        # the file name of training set
    val_split = "val_LCTSC"
    test_split = "test_LCTSCC3"           # the file name of testing set
    crop = None              # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "TransFuse"

class Config_VESSEL12:
    data_path = "../../dataset/CT/"
    data_subpath = "../../dataset/CT/VESSEL12/"
    save_path = "./checkpoints/CT/VESSEL12/"
    result_path = "./result/CT/VESSEL12/"
    tensorboard_path = "./tensorboard/CT/VESSEL12/"
    load_path = "./xxxx"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 200                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train_VESSEL12"        # the file name of training set
    val_split = "val_VESSEL12"
    test_split = "test_VESSEL12"           # the file name of testing set
    crop = None              # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "TransFuse"

class Config_ATM:
    data_path = "../../dataset/CT/"
    data_subpath = "../../dataset/CT/ATM/"
    save_path = "./checkpoints/CT/ATM/"
    result_path = "./result/CT/ATM/"
    tensorboard_path = "./tensorboard/CT/ATM/"
    load_path = "./xxxx"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 200                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train_ATM"        # the file name of training set
    val_split = "val_ATM"
    test_split = "test_ATM"           # the file name of testing set
    crop = None              # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "TransFuse"

class Config_INSTANCE:
    data_path = "../../dataset/CT/"
    data_subpath = "../../dataset/CT/INSTANCE/"
    save_path = "./checkpoints/CT/INSTANCE/"
    result_path = "./result/CT/INSTANCE/"
    tensorboard_path = "./tensorboard/CT/INSTANCE/"
    load_path = "./xxxx"
    visual_result_path = "./result/CT/INSTANCE/"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "all_train_INSTANCE_C1"        # the file name of training set
    val_split = "all_val_INSTANCE_C1"
    test_split = "all_test_INSTANCE_C1"           # the file name of testing set
    crop = None              # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "TransFuse"

# -------------------------------------------------------------------------------------------------
class Config_TN3K:
    data_path = "../../dataset/SAMUS/" 
    data_subpath = "../../dataset/SAMUS/TN3K/" 
    save_path = "./checkpoints/TN3K/"
    result_path = "./result/TN3K/"
    tensorboard_path = "./tensorboard/TN3K/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 800                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train-ThyroidNodule-TN3K"  # the file name of training set
    val_split = "val-ThyroidNodule-TN3K"     # the file name of testing set
    test_split = "test-ThyroidNodule-TN3K"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_ThyroidNodule:
    # This dataset is for thyroid nodule segmentation
    data_path = "../../dataset/SAMUS/"  
    save_path = "./checkpoints/ThyroidNodule/"
    result_path = "./result/ThyroidNodule/"
    tensorboard_path = "./tensorboard/ThyroidNodule/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                                 # number of data loading workers (default: 8)
    epochs = 400                                # number of total epochs to run (default: 400)
    batch_size = 8                              # batch size (default: 4)
    learning_rate = 1e-4                        # iniial learning rate (default: 0.001)
    momentum = 0.9                              # momntum
    classes = 2                                 # thenumber of classes (background + foreground)
    img_size = 256                              # theinput size of model
    train_split = "train-ThyroidNodule"    # the file name of training set
    val_split = "val-ThyroidNodule"        # the file name of testing set
    test_split = "test-ThyroidNodule-TN3K"      # the file name of testing set
    crop = None                                 # the cropped image size
    eval_freq = 1                               # the frequency of evaluate the model
    save_freq = 2000                            # the frequency of saving the model
    device = "cuda"                             # training device, cpu or cuda
    cuda = "on"                                 # switch on/off cuda option (default: off)
    gray = "yes"                                # the type of input image
    img_channel = 1                             # the channel of input image
    eval_mode = "mask_slice"                         # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"
    data_subpath = "../../dataset/SAMUS/BUSI/"   
    save_path = "./checkpoints/BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_Fascicle:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"  # 
    data_subpath = "../../dataset/SAMUS/Fascicle/"  # KvasirSeg or PolypSeg
    save_path = "./checkpoints/Fascicle/"
    result_path = "./result/Fascicle/"
    tensorboard_path = "./tensorboard/Fascicle/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Fascicle-FALLMUDRyan"   # the file name of training set
    val_split = "val-Fascicle-FALLMUDRyan"       # the file name of testing set
    test_split = "test-Fascicle-FALLMUDNeil"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_Aponeurosis:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"  #
    data_subpath = "../../dataset/SAMUS/Aponeurosis/" 
    save_path = "./checkpoints/Aponeurosis/"
    result_path = "./result/Aponeurosis/"
    tensorboard_path = "./tensorboard/Aponeurosis/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Aponeurosis-FALLMUDRyan"   # the file name of training set
    val_split = "val-Aponeurosis-FALLMUDRyan"       # the file name of testing set
    test_split = "test-Aponeurosis-FALLMUDNeil"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"  # 
    data_subpath = "../../dataset/SAMUS/CAMUS/" 
    save_path = "./checkpoints/CAMUS/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 4                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-EchocardiographyLA-CAMUS"   # the file name of training set
    val_split = "val-EchocardiographyLA-CAMUS"       # the file name of testing set
    test_split = "test-Echocardiography-Private"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camusmulti"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"



# ==================================================================================================
def get_config(task="Synapse"):
    if task == "SAMUS":
        return Config_SAMUS()
    elif task == "TN3K":
        return Config_TN3K()
    elif task == "ThyroidNodule":
        return Config_ThyroidNodule()
    elif task == "BUSI":
        return Config_BUSI()
    elif task == "Fascicle":
        return Config_Fascicle()
    elif task == "Aponeurosis":
        return Config_Aponeurosis()
    elif task == "CAMUS":
        return Config_CAMUS()
    elif task == "CT":
        return Config_CT()
    elif task == "CT5M":
        return Config_CT5M()
    elif task == "COVID19":
        return Config_COVID19()
    elif task == "FUMPE":
        return Config_FUMPE()
    elif task == "LCTSC":
        return Config_LCTSC()
    elif task == "VESSEL12":
        return Config_VESSEL12()
    elif task == "ATM":
        return Config_ATM()
    elif task == "INSTANCE":
        return Config_INSTANCE()
    elif task == "WORD":
        return Config_WORD()
    else:
        assert("We do not have the related dataset, please choose another task.")