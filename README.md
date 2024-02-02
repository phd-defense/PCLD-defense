# Drawing as Adversarial Manipulation Filter
Anonymous Author 1, Anonymous Author 2

## Abstract
Machine learning in general and computer vision models in particular are vulnerable to adversarial manipulation. Adversarial training and input transformations are two of the most prominent techniques for building adversary-resilient models. However, adversarial training faces challenges generalizing to unseen attacks and data, particularly in high-dimensional environments, while input transformations are ineffective against large perturbations. Painting algorithms try to capture the essential visual elements of an image and thereby are effective filters of adversarial perturbations. In this article, we show that the painting granularity and the magnitude of perturbations required to produce an adversarial effect are correlated. We use this observation for adversarial training of an ensemble classifier that analyses the painting process of an input image. This approach robustly addresses substantial perturbations and demonstrates generalizability across multiple attack methods.

|                       |                                        𝜀 = 0                                        |                                       𝜀 = 12                                        |                                        𝜀 = 24                                        |                                        𝜀 = 36                                        |                                        𝜀 = 51                                        |
|-----------------------|:------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|
| Input Image (`t = ∞`) | ![Image](./paper_results/drawing_process_example/original_n02101388_21983/eps_0.png) | ![Image](./paper_results/drawing_process_example/original_n02101388_21983/eps_12.png) | ![Image](./paper_results/drawing_process_example/original_n02101388_21983/eps_24.png) | ![Image](./paper_results/drawing_process_example/original_n02101388_21983/eps_36.png) | ![Image](./paper_results/drawing_process_example/original_n02101388_21983/eps_51.png) |
| Painting              |   ![Demo](./paper_results/drawing_process_example/demos_n02101388_21983/eps_0.gif)   |  ![Demo](./paper_results/drawing_process_example/demos_n02101388_21983/eps_12.gif)   |   ![Demo](./paper_results/drawing_process_example/demos_n02101388_21983/eps_24.gif)   |   ![Demo](./paper_results/drawing_process_example/demos_n02101388_21983/eps_36.gif)   |   ![Demo](./paper_results/drawing_process_example/demos_n02101388_21983/eps_51.gif)   |



## Install requirements
```
$ pip install -r requirements.txt
```
## Setup
### Setup 1: Download the dataset
#### You can use the exact dataset described in the paper by running the [notebook](./setup_1_subset_of_imagenet_downloader/Get Subset of ImageNet we Used in Paper.ipynb). This will create a new dataset in [subset_of_imagenet](./resources/datasets/subset_of_imagenet) folder and a small sample in [subset_of_imagenet_sample](./resources/datasets/subset_of_imagenet_sample) folder.
####

### Setup 2: Download the pretrained painter
#### To be able to generate paints given an image you should download the pretrained painter and the renderer models from inside [LearningToPaint](./LearningToPaint) folder.
####


## Phase 1 - Evaluating ph victim model
### 1. Generate paints from the benign dataset
```
$ python --experiment_type defend_drawing_agent --experiment_name ph2d_model --ds_local_path
./phd-defense/resources/datasets/subset_of_imagenet
--save_every
"200,700,1200,1700,2200,2700,3200,3700,4200,4700,5200" 
```
## Training h<sub>2</sub> (trained on paints and images) victim classifier
```
$ python 
```






[//]: # ()
[//]: # ()
[//]: # (![aimeos-frontend]&#40;https://user-images.githubusercontent.com/8647429/212348410-55cbaa00-722a-4a30-8b57-da9e173e0675.jpg&#41;)

[//]: # ()
[//]: # (## Table Of Content)

[//]: # ()
[//]: # (- [Installation]&#40;#installation&#41;)

[//]: # (    - [Composer]&#40;#composer&#41;)

[//]: # (    - [DDev or Colima]&#40;#ddev&#41;)

[//]: # (    - [TER]&#40;#ter-extension&#41;)

[//]: # (- [TYPO3 setup]&#40;#typo3-setup&#41;)

[//]: # (    - [Database setup]&#40;#database-setup&#41;)

[//]: # (    - [Security]&#40;#security&#41;)

[//]: # (- [Page setup]&#40;#page-setup&#41;)

[//]: # (    - [Download the Aimeos Page Tree t3d file]&#40;#download-the-aimeos-page-tree-t3d-file&#41;)

[//]: # (    - [Go to the Import View]&#40;#go-to-the-import-view&#41;)

[//]: # (    - [Upload the page tree file]&#40;#upload-the-page-tree-file&#41;)

[//]: # (    - [Go to the import view]&#40;#go-to-the-import-view&#41;)

[//]: # (    - [Import the page tree]&#40;#import-the-page-tree&#41;)

[//]: # (    - [SEO-friendly URLs]&#40;#seo-friendly-urls&#41;)

[//]: # (- [License]&#40;#license&#41;)

[//]: # (- [Links]&#40;#links&#41;)

[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # (This document is for the latest Aimeos TYPO3 **22.10 release and later**.)

[//]: # ()
[//]: # (- LTS release: 23.10 &#40;TYPO3 12 LTS&#41;)

[//]: # (- Old LTS release: 22.10 &#40;TYPO3 11 LTS&#41;)

[//]: # ()
[//]: # (### Composer)

[//]: # ()
[//]: # (**Note:** composer 2.1+ is required!)

[//]: # ()
[//]: # (The latest TYPO3 version can be installed via composer. This is especially useful, if you want to create new TYPO3 installations automatically or play with the latest code. You need to install the composer package first, if it isn't already available:)

[//]: # ()
[//]: # (```bash)

[//]: # (php -r "readfile&#40;'https://getcomposer.org/installer'&#41;;" | php -- --filename=composer)

[//]: # (```)

[//]: # ()
[//]: # (To install the TYPO3 base distribution first, execute this command:)

[//]: # ()
[//]: # (```bash)

[//]: # (composer create-project typo3/cms-base-distribution myshop)

[//]: # (# or install a specific TYPO3 version:)

[//]: # (composer create-project "typo3/cms-base-distribution:^12" myshop)

[//]: # (```)

[//]: # ()
[//]: # (It will install TYPO3 into the `./myshop/` directory. Change into the directory and install TYPO3 as usual:)

[//]: # ()
[//]: # (```bash)

[//]: # (cd ./myshop)

[//]: # (touch public/FIRST_INSTALL)

[//]: # (```)

[//]: # ()
[//]: # (Open the TYPO3 URL in your browser and follow the setup steps. Afterwards, install the Aimeos extension using:)

[//]: # ()
[//]: # (```bash)

[//]: # (composer req -W aimeos/aimeos-typo3:~23.7)

[//]: # (```)

[//]: # ()
[//]: # (If composer complains that one or more packages can't be installed because the required minimum stability isn't met, add this to your `composer.json`:)

[//]: # ()
[//]: # (```json)

[//]: # ("minimum-stability": "dev",)

[//]: # ("prefer-stable": true,)

[//]: # (```)

[//]: # ()
[//]: # (If you want a more or less working installation out of the box for new installations, you can install the Bootstrap package too:)

[//]: # ()
[//]: # (```bash)

[//]: # (composer req bk2k/bootstrap-package)

[//]: # (```)

[//]: # ()
[//]: # (***Note***: Remember to create a root page and a root template, which includes the Bootstrap package templates! &#40;See also below.&#41;)

[//]: # ()
[//]: # (Finally, depending on your TYPO3 version, run the following commands from your installation directory:)

[//]: # ()
[//]: # (**For TYPO3 11+:**)

[//]: # ()
[//]: # (```bash)

[//]: # (php ./vendor/bin/typo3 extension:setup)

[//]: # (php ./vendor/bin/typo3 aimeos:setup --option=setup/default/demo:1)

[//]: # (```)

[//]: # ()
[//]: # (If you don't want to add the Aimeos demo data, you should remove `--option=setup/default/demo:1` from the Aimeos setup command.)

[//]: # ()
[//]: # (**For TYPO3 10:**)

[//]: # ()
[//]: # (```bash)

[//]: # (php ./vendor/bin/typo3 extension:activate scheduler)

[//]: # (php ./vendor/bin/typo3 extension:activate aimeos)

[//]: # (```)

[//]: # ()
[//]: # (If you experience any errors with the database, please check the [Database Setup]&#40;#database-setup&#41; section below.)

[//]: # ()
[//]: # (Please keep on reading below the "TER Extension" installation section!)

[//]: # ()
[//]: # (### DDev)

[//]: # ()
[//]: # (*Note:* Installation instructions for TYPO3 with `ddev` or `Colima` can be found here:)

[//]: # ([TYPO3 with ddev or colima]&#40;https://ddev.readthedocs.io/en/latest/users/quickstart/&#41;)

[//]: # ()
[//]: # (### TER Extension)

[//]: # ()
[//]: # (If you want to install Aimeos into a traditionally installed TYPO3 &#40;"legacy installation"&#41;, the [Aimeos extension from the TER]&#40;https://typo3.org/extensions/repository/view/aimeos&#41; is recommended. You can download and install it directly from the Extension Manager of your TYPO3 instance.)

[//]: # ()
[//]: # (* Log into the TYPO3 backend)

[//]: # (* Click on "Admin Tools::Extensions" in the left navigation)

[//]: # (* Click the icon with the little plus sign left from the Aimeos list entry)

[//]: # ()
[//]: # (![Install Aimeos TYPO3 extension]&#40;https://user-images.githubusercontent.com/213803/211545083-d0820b63-26f2-453e-877f-ecd5ec128713.jpg&#41;)

[//]: # ()
[//]: # (Afterwards, you have to execute the update script of the extension to create the required database structure:)

[//]: # ()
[//]: # (* Click on "Admin Tools::Upgrade")

[//]: # (* Click "Run Upgrade Wizard" in the "Upgrade Wizard" tile)

[//]: # (* Click "Execute")

[//]: # ()
[//]: # (![Execute update script]&#40;https://user-images.githubusercontent.com/213803/211545122-8fd94abd-78b2-47ad-ad3c-1ef1b9c052b4.jpg&#41;)

[//]: # ()
[//]: # (#### Aimeos Distribution)

[//]: # ()
[//]: # (For new TYPO3 installations, there is a 1-click [Aimeos distribution]&#40;https://typo3.org/extensions/repository/view/aimeos_dist&#41; available, too. Choose the Aimeos distribution from the list of available distributions in the Extension Manager and you will get a completely set up shop system including demo data for a quick start.)

[//]: # ()
[//]: # (## TYPO3 Setup)

[//]: # ()
[//]: # (Setup TYPO3 by creating a `FIRST_INSTALL` file in the `./public` directory:)

[//]: # ()
[//]: # (```bash)

[//]: # (touch public/FIRST_INSTALL)

[//]: # (```)

[//]: # ()
[//]: # (Open the URL of your installation in the browser and follow the steps in the TYPO3 setup scripts.)

[//]: # ()
[//]: # (### Database Setup)

[//]: # ()
[//]: # (If you use MySQL < 5.7.8, you have to use `utf8` and `utf8_unicode_ci` instead because those MySQL versions can't handle the long indexes created by `utf8mb4` &#40;up to four bytes per character&#41; and you will get errors like)

[//]: # ()
[//]: # (```)

[//]: # (1071 Specified key was too long; max key length is 767 bytes)

[//]: # (```)

[//]: # ()
[//]: # (To avoid that, change your database settings in your `./typo3conf/LocalConfiguration.php` to:)

[//]: # ()
[//]: # (```php)

[//]: # (    'DB' => [)

[//]: # (        'Connections' => [)

[//]: # (            'Default' => [)

[//]: # (                'tableoptions' => [)

[//]: # (                    'charset' => 'utf8',)

[//]: # (                    'collate' => 'utf8_unicode_ci',)

[//]: # (                ],)

[//]: # (                // ...)

[//]: # (            ],)

[//]: # (        ],)

[//]: # (    ],)

[//]: # (```)

[//]: # ()
[//]: # (### Security)

[//]: # ()
[//]: # (Since **TYPO3 9.5.14+** implements **SameSite cookie handling** and restricts when browsers send cookies to your site. This is a problem when customers are redirected from external payment provider domain. Then, there's no session available on the confirmation page. To circumvent that problem, you need to set the configuration option `cookieSameSite` to `none` in your `./typo3conf/LocalConfiguration.php`:)

[//]: # ()
[//]: # (```php)

[//]: # (    'FE' => [)

[//]: # (        'cookieSameSite' => 'none')

[//]: # (    ])

[//]: # (```)

[//]: # ()
[//]: # (## Site Setup)

[//]: # ()
[//]: # (TYPO3 10+ requires a site configuration which you have to add in "Site Management" > "Sites" available in the left navigation. When creating a root page &#40;a page with a globe icon&#41;, a basic site configuration is automatically created &#40;see below at [Go to the Import View]&#40;#go-to-the-import-view&#41;&#41;.)

[//]: # ()
[//]: # (## Page Setup)

[//]: # ()
[//]: # (### Download the Aimeos Page Tree t3d file)

[//]: # ()
[//]: # (The page setup for an Aimeos web shop is easy, if you import the example page tree for TYPO3 10/11. You can download the version you need from here:)

[//]: # ()
[//]: # (* [23.4+ page tree]&#40;https://aimeos.org/fileadmin/download/Aimeos-pages_2023.04.t3d&#41; and later)

[//]: # (* [22.10 page tree]&#40;https://aimeos.org/fileadmin/download/Aimeos-pages_2022.10.t3d&#41;)

[//]: # (* [21.10 page tree]&#40;https://aimeos.org/fileadmin/download/Aimeos-pages_21.10.t3d&#41;)

[//]: # ()
[//]: # (**Note:** The Aimeos layout expects [Bootstrap]&#40;https://getbootstrap.com&#41; providing the grid layout!)

[//]: # ()
[//]: # (In order to upload and install the file, follow the following steps:)

[//]: # ()
[//]: # (### Go to the Import View)

[//]: # ()
[//]: # (**Note:** It is recommended to import the Aimeos page tree to a page that is defined as "root page". To create a root page, simply create a new page and, in the "Edit page properties", activate the "Use as Root Page" option under "Behaviour". The icon of the root page will change to a globe. This will also create a basic site configuration. Don't forget to also create a typoscript root template and include the bootstrap templates with it!)

[//]: # ()
[//]: # (![Create a root page]&#40;https://user-images.githubusercontent.com/213803/211549273-1d3883dd-710c-4e27-8dbb-3de6e45680d7.jpg&#41;)

[//]: # ()
[//]: # (* In "Web::Page", right-click on the root page &#40;the one with the globe&#41;)

[//]: # (* Click on "More options...")

[//]: # (* Click on "Import")

[//]: # ()
[//]: # (![Go to the import view]&#40;https://user-images.githubusercontent.com/213803/211550212-df6daa73-74cd-459e-8d25-a56c413c175d.jpg&#41;)

[//]: # ()
[//]: # (### Upload the page tree file)

[//]: # ()
[//]: # (* In the page import dialog)

[//]: # (* Select the "Upload" tab &#40;2nd one&#41;)

[//]: # (* Click on the "Select" dialog)

[//]: # (* Choose the T3D file you've downloaded)

[//]: # (* Press the "Upload files" button)

[//]: # ()
[//]: # (![Upload the page tree file]&#40;https://user-images.githubusercontent.com/8647429/212347778-17238e05-7494-4413-adb3-a54b2b524e05.png&#41;)

[//]: # ()
[//]: # (### Import the page tree)

[//]: # ()
[//]: # (* In Import / Export view)

[//]: # (* Select the uploaded file from the drop-down menu)

[//]: # (* Click on the "Preview" button)

[//]: # (* The pages that will be imported are shown below)

[//]: # (* Click on the "Import" button that has appeared)

[//]: # (* Confirm to import the pages)

[//]: # ()
[//]: # (![Import the uploaded page tree file]&#40;https://user-images.githubusercontent.com/8647429/212348040-c3e10b60-5579-4d1b-becc-72548826c6db.png&#41;)

[//]: # ()
[//]: # (Now you have a new page "Shop" in your page tree including all required sub-pages.)

[//]: # ()
[//]: # (### SEO-friendly URLs)

[//]: # ()
[//]: # (TYPO3 9.5 and later can create SEO friendly URLs if you add the rules to the site config:)

[//]: # ([https://aimeos.org/docs/latest/typo3/setup/#seo-urls]&#40;https://aimeos.org/docs/latest/typo3/setup/#seo-urls&#41;)

[//]: # ()
[//]: # (## License)

[//]: # ()
[//]: # (The Aimeos TYPO3 extension is licensed under the terms of the GPL Open Source)

[//]: # (license and is available for free.)

[//]: # ()
[//]: # (## Links)

[//]: # ()
[//]: # (* [Web site]&#40;https://aimeos.org/integrations/typo3-shop-extension/&#41;)

[//]: # (* [Documentation]&#40;https://aimeos.org/docs/TYPO3&#41;)

[//]: # (* [Forum]&#40;https://aimeos.org/help/typo3-extension-f16/&#41;)

[//]: # (* [Issue tracker]&#40;https://github.com/aimeos/aimeos-typo3/issues&#41;)

[//]: # (* [Source code]&#40;https://github.com/aimeos/aimeos-typo3&#41;)
