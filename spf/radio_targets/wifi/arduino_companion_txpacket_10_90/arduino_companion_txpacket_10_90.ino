void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  Serial1.begin(115200);
}

void set_tx(int load, int attenuation) {
  Serial1.println("cmdstop");
  Serial1.println("cbw40m_en 0");
  Serial1.println("tx_contin_en 0");
  Serial1.println("tx_cbw40m_en 0");
  Serial1.println("RFChannelSel 1 0");
  Serial1.println("cmdstop");
  Serial1.println("FillTxPacket 655410 4 0 0 0 0 1 2 3 4 5 6");
  Serial1.print("target_power_backoff");
  Serial1.println(attenuation);
  if (load==90) { 
    Serial1.println("WifiTxStart 655360 0 67 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  } else if (load==50) {
    Serial1.println("WifiTxStart 655360 0 554 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  } else if (load==10) {
    Serial1.println("WifiTxStart 655360 0 5354 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  } else {
    //50 by default?
    Serial1.println("WifiTxStart 655360 0 554 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  }
}

void loop() {
   long load = random(0, 2); // load
   if (load==0) {
    load=10;
   } else if (load==1) {
    load=50;
   } else { //else if (load==2) {
    load=90;
   }
   long attenuation = random(30, 100); /// attenuation  
   set_tx(load, attenuation);
   digitalWrite(LED_BUILTIN, LOW);  // turn the LED on
   delay(1000*120); // two minutes
   digitalWrite(LED_BUILTIN, HIGH);  // turn the LED  off
   delay(1000);  
}
