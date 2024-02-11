void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  Serial1.begin(115200);

   Serial1.println("cbw40m_en 0");
   Serial1.println("tx_contin_en 1");
   Serial1.println("esp_tx 1 0 0");
   digitalWrite(LED_BUILTIN, LOW);  // turn the LED on (HIGH is the voltage level)
   delay(120000);  
}

void loop() {
}
