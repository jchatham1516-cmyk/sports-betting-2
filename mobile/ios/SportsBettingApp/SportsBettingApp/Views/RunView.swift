import SwiftUI

struct RunView: View {
    @StateObject private var viewModel = RunViewModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("Sport") {
                    Picker("Sport", selection: $viewModel.selectedSport) {
                        ForEach(Sport.allCases) { sport in
                            Text(sport.rawValue.uppercased()).tag(sport)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                Section("Game Date") {
                    DatePicker("", selection: $viewModel.selectedDate, displayedComponents: .date)
                        .datePickerStyle(.compact)
                }

                Section {
                    Button {
                        Task {
                            await viewModel.createRun()
                        }
                    } label: {
                        HStack {
                            Spacer()
                            if viewModel.isLoading {
                                ProgressView()
                            } else {
                                Text("Run Model")
                                    .fontWeight(.semibold)
                            }
                            Spacer()
                        }
                    }
                    .disabled(viewModel.isLoading)
                }
            }
            .navigationTitle("Run Model")
            .alert("Something went wrong", isPresented: Binding(
                get: { viewModel.errorMessage != nil },
                set: { _ in viewModel.errorMessage = nil }
            )) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(viewModel.errorMessage ?? "Please try again.")
            }
            .navigationDestination(item: $viewModel.runResponse) { response in
                RunDetailView(viewModel: RunDetailViewModel(runId: response.runId))
            }
        }
    }
}

#Preview {
    RunView()
}
