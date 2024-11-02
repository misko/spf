/*

  this project also realess in GitHub:
  https://github.com/Heltec-Aaron-Lee/WiFi_Kit_series
*/

#include "heltec.h"
#include "images.h"

//#define BAND    868E6  //you can set band here directly,e.g. 868E6,915E6
#define BAND    915E6  //you can set band here directly,e.g. 868E6,915E6
#define SEGMENT_LENGTH 10000
#define MSGLEN 32
uint8_t buffer[256];
uint32_t broadcast_time=10;
uint32_t sleep_time=10;
unsigned long start_time=0;

float avg_send_time = 50.0;
float target_duty = 0.5;
unsigned int counter = 0;
String rssi = "RSSI --";
String packSize = "--";
String packet ;

void logo()
{
  Heltec.display->clear();
  Heltec.display->drawXbm(0,5,logo_width,logo_height,logo_bits);
  Heltec.display->display();
}

void setup()
{
   //WIFI Kit series V1 not support Vext control
  Heltec.begin(true /*DisplayEnable Enable*/, true /*Heltec.Heltec.Heltec.LoRa Disable*/, true /*Serial Enable*/, true /*PABOOST Enable*/, BAND /*long BAND*/);
 
  Heltec.display->init();
  Heltec.display->flipScreenVertically();  
  Heltec.display->setFont(ArialMT_Plain_10);
  logo();
  delay(1500);
  Heltec.display->clear();
  
  Heltec.display->drawString(0, 0, "Heltec.LoRa Initial success!");
  Heltec.display->display();
  delay(1000);
  LoRa.setPreambleLength(6);
  LoRa.disableCrc();

  //LoRa.setTxPower(20,RF_PACONFIG_PASELECT_PABOOST);

  for (int i=0; i<MSGLEN; i++){
    buffer[i]=random(256);
  }
}

void loop()
{
  if (counter%1==0) {
    Heltec.display->clear();
    Heltec.display->setTextAlignment(TEXT_ALIGN_LEFT);
    Heltec.display->setFont(ArialMT_Plain_10);
    
    Heltec.display->drawString(0, 0, "Sending packet: ");
    Heltec.display->drawString(90, 0, String(broadcast_time));
    Heltec.display->display();
  }



  if (millis() - start_time > SEGMENT_LENGTH) {
    sleep_time = random(5,16);
    broadcast_time = random(3,40);
    LoRa.setTxPower(random(8,20),RF_PACONFIG_PASELECT_PABOOST);
    start_time=millis();
  }

  // send packet
  LoRa.beginPacket();
  
  LoRa.write(buffer,MSGLEN);


  LoRa.endPacket(true);
  digitalWrite(LED, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(broadcast_time);

  LoRa.idle();
  digitalWrite(LED, LOW);    // turn the LED off by making the voltage LOW
  delay(sleep_time);
}

