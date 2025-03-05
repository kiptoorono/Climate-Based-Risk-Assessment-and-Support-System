import os
import requests
import re 

class Data_downloader:
    def __init__(self,directory_url,download_path):
        self.directory_url=directory_url
        self.download_path= download_path
    
    def fetch_url(self,url):
        response= requests.get(url)
        if response.status_code ==200:
            relative_links=re.findall(r'href=[\'"]?([^\'" >]+)',response.text)
            print(relative_links)
            link_lists=[]
            for link in relative_links:
                if not re.match((r"^[^a-zA-z0-0]+:https://"), link):
                    link_lists.append(link)
            return [link for link in link_lists if not link.startswith("/public") and not link.startswith("?")]
        
        else:
             print(f'Could not fetch link from:{url}')
    
    def file_downloader(self,current_url,current_path):
        #invoke fetch Url method to get relative links 
       
        url_list =self.fetch_url(current_url);
        
        for link in url_list:
            new_url = f"{current_url}{link}"
            #check if relative link
            if link.endswith('/'):
                new_path=os.path.join(current_path,link)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                self.file_downloader(new_url,new_path)
                # if it is absolute(file) create path & download 
            else:
                file_path = os.path.join(current_path, link)
                file_path = file_path.replace('\\', '/')
                #check if file is already exists
                if os.path.exists(file_path):
                    print(f'File already exists: {file_path}')
                    continue
            
                try:
                    response= requests.get(new_url)
                    if response.status_code==200:
                        with open(file_path,'wb') as file:
                            file.write(response.content)
                            print(f'downloaded{new_url}: status code{response.status_codes}')
                    else:
                        print(f'failed to download from{new_url} statuscode:{response.status_codes}')
                except Exception as e:
                    print(f'Exception Occured while downloading {new_url}:{e}')
    #method to initiate download
    def start_downloading(self):
        self.file_downloader(self.directory_url,self.download_path)
    
directory_url= 'https://gws-access.jasmin.ac.uk/public/tamsat/soil_moisture/data/v2.3.0/dekadal/'
download_path = 'E:\Agriculture project\Data\Soil Moisture'
downloader=Data_downloader(directory_url,download_path)
downloader.start_downloading()