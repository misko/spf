# ESP32 WROOM 32U radio target

Instead of a constant tone target we can use a ESP32 in FCC test mode to emit continous wifi data on a specific channel and duty cycle.

## Construction

![ESP32-WROOM x 2](esp32_wroom_32u.jpg) [amazon link](https://www.amazon.com/gp/product/B09Z7MWHSD/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&th=1)

![ipex->SMA antenna](ipex_sma_antenna.jpg) [amazon link](https://www.amazon.com/gp/product/B095JTW6XM/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)

![USB to serial](usb_to_serial.jpg) [amazon link](https://www.amazon.com/gp/product/B00IJXZQ7C/ref=ppx_yo_dt_b_search_asin_image?ie=UTF8&psc=1)

## Firmware / ESPTestTool

Download official windows ESPTestTool [link](https://www.espressif.com/en/support/download/other-tools). 

Connect (over USB) to the device, load firmware SP32_RFTest_190 , load to flash, then reboot

![Load to flash](load_flash_0.png)

![Load to flash](load_flash_1.png)

## Intercepting commands 

Windows -> (USB) -> ESP32
MacOSX -> (USB) -> USB/Serial converted -> (TX/RX/GND) -> ESP32 (TX/RX/GND)

Issue commands for testing and intercept using minicom on laptop,

```
#TX continous chan1
cbw40m_en 0                                                                                                    
tx_contin_en 1                                                                                                 
esp_tx 1 0 0  
                                                                                                 
cmdstop   #STOP                                              
tx_contin_en 0              
               
#TX continuous chan2                                                                                            
cbw40m_en 0                                                                                                    
tx_contin_en 1                                                                                                 
esp_tx 2 0 0      
                                                                                             
cmdstop       #STOP                                                         
tx_contin_en 0         
                     

#TX packet                                                                                                   
cbw40m_en 0                                                                                                    
tx_contin_en 0
esp_tx 1 0 0

cmdstop #STOP

#TX tone
cbw40m_en 0
wifiscwout 1 1 0

wifiscwout 0 1 0 #STOP
```

## Running with companion arduino

[Running FCC test mode](https://youtu.be/tJOxaxYn43A)

[FCC blink](ardunio_companion_blink_FCC_code)

[FCC test](ardunio_companion)