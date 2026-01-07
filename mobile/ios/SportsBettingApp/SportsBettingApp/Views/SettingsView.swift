import SwiftUI

struct SettingsView: View {
    var body: some View {
        NavigationStack {
            List {
                Section("Environment") {
                    LabeledContent("Base URL") {
                        Text(AppConfig.baseURL.absoluteString)
                            .font(.footnote)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.trailing)
                    }
                    LabeledContent("Version") {
                        Text(AppConfig.appVersion)
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                }

                Section("Disclaimer") {
                    Text("This app provides sports betting insights for informational purposes only and does not place wagers or guarantee outcomes.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                    NavigationLink("Read full disclaimer") {
                        DisclaimerView()
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}

#Preview {
    SettingsView()
}
