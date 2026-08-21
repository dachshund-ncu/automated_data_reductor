# 🌌 Streamlit NCU 32m Radio Telescope Data Reductor
An intuitive Streamlit application designed to automate the reduction of spectral data acquired from the 32-meter Nicolaus Copernicus University (NCU) Radio Telescope. This tool streamlines the post-observation processing, allowing users to effortlessly reduce their spectral data and download the resulting FITS files.

# 🚀 Getting Started
Follow these steps to get the Streamlit NCU Data Reductor up and running on your local machine.

## Prerequisites
Before you begin, ensure you have the following installed:

- Python 3.11 - 3.13
- [UV](https://docs.astral.sh/uv/)

## Installation
#### Clone the repository:

```bash
git https://github.com/dachshund-ncu/automated_data_reductor.git
cd automated_data_reductor
```

#### Install application:
```bash
uv python pin 3.12
uv sync
uv tool install .
```

## Running the App
Once the dependencies are installed, you can run the Streamlit application by typing anywhere:

```bash
art4r
```
The app will automatically open in your web browser (usually at http://localhost:8501).

# 👨‍💻 Usage
### 1. Upload Your Data:

On the Streamlit interface, you will find an option to upload your raw spectral data files from the NCU 32m Radio Telescope. 
The expected format is typically .tar.bz2
You will have an opportunity to choose parameters:
- BBC for LHC
- BBC for RHC
- caltabs usage (use caltabs or not)
- reduction mode (frequency-switch or on-off)

### 2. Initiate Reduction: 

Click "submit" button. Processing might take a while. The progress bar will keep you informed about data reduction progress.


### 3. Download Results: 

Once the reduction is complete, a download link will appear, allowing you to save the processed .fits file(s) to your local machine.

## 🛠️ Technologies Used
Python
Streamlit - For building the interactive web application.
NumPy - For numerical operations.
Astropy - For handling FITS files and astronomical calculations.
Tensorflow - for models to recognize RFI

# 📄 License
This project is licensed under the MIT License.

# 📞 Contact
If you have any questions or need further assistance, please open an issue on the GitHub repository or contact:
```
Michał Durjasz
ex Nicolaus Copernicus University staff member
email: md@astro.umk.pl
```
