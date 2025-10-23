import os
import requests

file_list  = [
    "SiCo30_01.txt", "GaCo11_01.txt", "SiCo27_01.txt", "SiCo14_01.txt", "SiCo17_01.txt",
    "GaCo02_01.txt", "GaCo13_01.txt", "GaCo17_01.txt", "GaCo08_01.txt", "SiCo18_01.txt",
    "SiCo06_01.txt", "GaCo12_01.txt", "SiCo03_01.txt", "GaCo06_01.txt", "SiCo19_01.txt",
    "GaCo04_01.txt", "GaCo22_01.txt", "SiCo04_01.txt", "SiCo25_01.txt", "GaCo16_01.txt",
    "SiCo10_01.txt", "SiCo20_01.txt", "SiCo21_01.txt", "GaCo09_01.txt", "SiCo13_01.txt",
    "SiCo09_01.txt", "GaCo01_01.txt", "GaCo05_01.txt", "SiCo26_01.txt", "GaCo10_01.txt",
    "SiCo12_01.txt", "SiCo29_01.txt", "SiCo15_01.txt", "SiCo11_01.txt", "SiCo28_01.txt",
    "SiCo07_01.txt", "SiCo24_01.txt", "SiCo01_01.txt", "GaCo03_01.txt", "GaCo15_01.txt",
    "SiPt34_01.txt", "GaPt13_01.txt", "GaPt14_01.txt", "SiPt13_01.txt", "SiPt38_01.txt",
    "SiPt21_01.txt", "SiPt02_01.txt", "SiPt07_01.txt", "GaPt03_01.txt", "GaPt30_01.txt",
    "SiPt36_01.txt", "SiPt28_01.txt", "GaPt07_01.txt", "GaPt19_01.txt", "SiPt04_01.txt",
    "SiPt18_01.txt", "GaPt23_01.txt", "GaPt15_01.txt", "GaPt20_01.txt", "GaPt12_01.txt",
    "SiPt39_01.txt", "SiPt32_01.txt", "SiPt30_01.txt", "GaPt22_01.txt", "SiPt40_01.txt",
    "GaPt05_01.txt", "GaPt33_01.txt", "SiPt33_01.txt", "SiPt10_01.txt", "GaPt08_01.txt",
    "GaPt24_01.txt", "SiPt05_01.txt", "GaPt04_01.txt", "SiPt24_01.txt", "SiPt35_01.txt",
    "SiPt23_01.txt", "SiPt08_01.txt", "SiPt25_01.txt", "SiPt12_01.txt", "GaPt27_01.txt"
]



base_url = "https://physionet.org/files/gaitpdb/1.0.0/"

dst_dir = r"Data/RawData/"
os.makedirs(dst_dir, exist_ok=True)

for fname in file_list:
    url = base_url + fname+ "?download"
    dst_path = os.path.join(dst_dir, fname)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {fname} from {url} to {dst_path}")
    except requests.HTTPError as e:
        print(f"Failed to download {fname}: {e}")
