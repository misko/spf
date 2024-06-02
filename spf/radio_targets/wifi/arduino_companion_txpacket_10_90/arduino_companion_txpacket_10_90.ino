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
  Serial1.print("target_power_backoff ");
  Serial1.println(attenuation);
  if (load==90) { 
    //Serial1.println("WifiTxStart 655360 0 67 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
    Serial1.println("WifiTxStart 655360 0 323 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  } else if (load==50) {
    Serial1.println("WifiTxStart 655360 0 554 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  } else if (load==35) {
    Serial1.println("WifiTxStart 655360 0 1527 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  } else if (load==25) {
    Serial1.println("WifiTxStart 655360 0 2500 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  } else if (load==10) {
    Serial1.println("WifiTxStart 655360 0 5354 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  } else {
    //50 by default?
    Serial1.println("WifiTxStart 655360 0 554 0 0 1 0"); // 90 -> 67 , 10 -> 5354 , 50 -> 554
  }
}

void loop() {
   long load = random(0, 4); // load
   if (load==0) {
    load=10;
   } else if (load==1) {
    load=50;
   } else if (load==2) {
    load=25;
   } else if (load==3) {
    load=35;
   } else { //else if (load==3) {
    load=90;
   }
   long attenuation = 50; // random(30, 30); /// attenuation  
   set_tx(load, attenuation);
   for (int idx=0; idx<30; idx++) { // 30 seconds in each state
    if (idx%5==0) {
      digitalWrite(LED_BUILTIN, LOW);  // turn the LED on
    } else {
      digitalWrite(LED_BUILTIN, HIGH);  // turn the LED on
    }
    delay(1000); 
   }
   digitalWrite(LED_BUILTIN, HIGH);  // turn the LED  off
   delay(1000);  
}
