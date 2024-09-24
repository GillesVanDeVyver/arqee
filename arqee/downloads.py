import arqee.CONST as CONST
import os
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download


def set_up_model(model_path,verbose=True):
    '''
    Unzips the model to the model directory
    :param model_path: str
        Path to the downloaded zip file
        The filename should be {model_name}.zip
    :param verbose: bool
        If True, print info to the standard output
    :return: None
    '''
    # check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist."
                                f" Please provide a valid path to the model zip file when running set_up_model.")
    if verbose:
        os.system(f"unzip -o {model_path} -d {CONST.MODEL_DIR}")
    else:
        # Redirecting standard output to /dev/null
        os.system(f"unzip -o {model_path} -d {CONST.MODEL_DIR} > /dev/null 2>&1")
    if verbose:
        filename = os.path.basename(model_path)
        print(f"Model {filename} set up successfully!")

def remove_model(model_name,verbose=True):
    '''
    Removes the model from the model directory
    :param model_name: str
        The model name of the model to remove
    :param verbose: bool
        If True, print info to the standard output
    :return: None
    '''
    model_path = os.path.join(CONST.MODEL_DIR, model_name)
    if os.path.exists(model_path):
        if verbose:
            print(f"Removing model {model_name} from {model_path}")
        os.system(f"rm -r {model_path}")
        if verbose:
            print(f"Model {model_name} removed successfully!")
    else:
        if verbose:
            print(f"Model {model_name} not found in {model_path}")

def remove_all_models(verbose=True):
    '''
    Removes all models from the model directory
    :param verbose: bool
        If True, print info to the standard output
    :return: None
    '''
    if verbose:
        print(f"Removing all models from {CONST.MODEL_DIR}")
    os.system(f"rm -r {CONST.MODEL_DIR}/*")
    if verbose:
        print(f"All models removed successfully!")


def get_model_download_link(model_name="mobilenetv2_regional_quality",verbose=True):
    '''
    Returns the download link for the model with the given backbone and print to the standard output if verbose is True
    Remarks:
    - The larger backbones give slightly better results
    - The models are only trained on labels where more than 50% of the region is inside the sector (according to the annotators).
    - The models look at the quality of the myocardium itself, not the endocardial border visibility.
    :param model_name: str
        The model name of the model to get a download link for. Supported models are:
        - 'mobilenetv2_regional_quality'
        - 'nnunet_hunt4_a2c_a4c'
        - 'nnunet_hunt4_alax'
    :param verbose: str
        If True, print instructions to the standard output
    :return: str
        The download link for the model with the given backbone
    '''
    supported_models = ['mobilenetv2_regional_quality', 'nnunet_hunt4_a2c_a4c', 'nnunet_hunt4_alax']
    if model_name in supported_models:
        if verbose:
            print(f"Download link for model with backbone {model_name} is {CONST.DOWNLOAD_LINKS[model_name]}")
        link=CONST.DOWNLOAD_LINKS[model_name]
    else:
        raise NotImplementedError(f"No download link for model with backbone {model_name} available."
                                  + f"Supported backbones are {supported_models}")
    return link

def download_file(url, filepath,verbose=True):
    '''
    Download the file from the given URL to the given filepath
    :param url: str
        The URL to download the file from
    :param filepath: str
        The filepath where the file will be saved
    :param verbose: str
        If True, print info and progress to the standard output
    :return: None
    '''
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        if verbose:
            print(f"Downloaded: {filepath}")
    else:
        if verbose:
            print(f"Failed to download: {url}")

def download_data_sample(download_folder,verbose=True,custom_data_loc=None):
    '''
    Download sample data of 1.7GB to the target directory
    The sample data contains .npy and .mp4 files of ultrasound cardiac data obtained from an E95 scanner.
    The data contains left A2C, A4C and ALAX views.
    :param download_folder: str
        The folder where the sample data will be downloaded to. A folder called 'sample_data' will be created in this
    :param verbose: bool
        If True, print info and progress to the standard output
    :param custom_data_loc: str
        If not None, the custom data location will be used instead of the default data location in CONST.py
    :return: None
    '''
    download_loc = os.path.join(download_folder, 'sample_data')
    # Create the download directory if it doesn't exist
    if not os.path.exists(download_loc):
        os.makedirs(download_loc)

    # GitHub API URL to list contents of a directory
    if custom_data_loc is None:
        api_url = CONST.DOWNLOAD_LINKS['sample_data']
    else:
        api_url = custom_data_loc
    if verbose:
        print(f"Fetching contents from: {api_url}")

    # Make a GET request to fetch the contents of the directory
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Iterate over each item in the directory
        for item in tqdm(response.json()):
            # Download only files, skip directories
            if item['type'] == 'file':
                # Construct the download URL for the file
                file_download_url = item['download_url']
                # Construct the filepath where the file will be saved
                filepath = os.path.join(download_loc, item['name'])
                # Download the file
                download_file(file_download_url, filepath,verbose=verbose)
    else:
        print(f"Failed to fetch contents: {api_url}")


def download_model_from_huggingface(repo_id, filename, model_dir=CONST.MODEL_DIR, verbose=True):
    """
    Downloads a model file from Hugging Face Hub directly
    :param repo_id: str
        The repository id on Hugging Face (e.g. 'username/repo_name')
    :param filename: str
        The filename of the model in the repository to download
    :param model_dir: str
        The directory where the model will be saved
    :param verbose: bool
        If True, print info to the standard output
    :return: str
                Returns the full path to the downloaded model file after successful download.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if verbose:
        print(f"Downloading {filename} from Hugging Face repository {repo_id}...")

    try:
        downloaded_file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=model_dir)
        if verbose:
            print(f"Model downloaded to {downloaded_file_path}")
        # return the location where the model was downloaded
        return downloaded_file_path
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise


def download_and_set_up_model(model_name, verbose=True):
    '''
    Downloads and sets up a model by fetching its download link, downloading it, and extracting it to the model directory.

    This function combines three key steps:
    1. Gets the download link for the specified model.
    2. Downloads the model from the Hugging Face repository.
    3. Unzips the downloaded file and sets it up in the designated model directory.

    :param model_name: str
        The name of the model to be downloaded and set up. Supported models are:
        - 'mobilenetv2_regional_quality'
        - 'nnunet_hunt4_a2c_a4c'
        - 'nnunet_hunt4_alax'
    :param verbose: bool
        If True, prints progress and information to the standard output.
    :return: None
    '''
    (repo_id, filename) = get_model_download_link(model_name, verbose=True)
    download_path = download_model_from_huggingface(repo_id, filename, verbose=verbose)
    set_up_model(download_path)
