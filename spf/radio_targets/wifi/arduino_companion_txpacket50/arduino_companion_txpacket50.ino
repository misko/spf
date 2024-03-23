void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  Serial1.begin(115200);
  Serial1.println("cmdstop");
  Serial1.println("cbw40m_en 0");
  Serial1.println("tx_contin_en 0");
  Serial1.println("tx_cbw40m_en 0");
  Serial1.println("RFChannelSel 1 0");
  Serial1.println("cmdstop");
  Serial1.println("FillTxPacket 655410 4 0 0 0 0 1 2 3 4 5 6");
  Serial1.println("target_power_backoff 30");
  Serial1.println("WifiTxStart 655360 0 554 0 0 1 0");
}

void loop() {
   digitalWrite(LED_BUILTIN, LOW);  // turn the LED on (HIGH is the voltage level)
   delay(2000);  
   digitalWrite(LED_BUILTIN, HIGH);  // turn the LED  on (HIGH is the voltage level)
   delay(1000);  
}
