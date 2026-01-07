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

                Section("Legal") {
                    NavigationLink("Disclaimer") {
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
