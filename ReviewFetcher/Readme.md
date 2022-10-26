# Review Fetcher Module
## _This module fetches the reviews of the specied App_.
## Step 1: How to Specify the App ##
You can specify the app for which the reviews will be extracted. In code block 5 of the _App_ReviewExtraction.ipynb_ file, you need to mention the package of the intented app. For example 
```sh
app_packages = [
  'com.skype.raider'
]
```
## Step 1: How to Find Package of the App ##
Go to google playstore and open the intended app. In the url you will find the  _id=xxx.yyy.zzz_. For example - In the figure you can find the id in the url like  _id=com.skype.raider_. You need to copy the _xxx.yyy.zzz_ i.e. _com.skype.raider_ and paste it to the mentioned block of the source code. 

![Alt text](ReviewFetching.png?raw=true "Title")
## Other Settings ##
By-default it fetches 1200 reviews and it is possible to mention the language and region from where the reviews are to be extracted (See the code block 6 in the _App_ReviewExtraction.ipynb_ file). 
```sh
info = app(ap, lang='en', country='us')
```
You can also sort the reviews by specifying different parameters used by Google Playstore like MOST_RELEVANT, NEWEST etc. See the code block 11 in _App_ReviewExtraction.ipynb_ file).
```sh
for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
```
