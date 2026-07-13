// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nemo_relay

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenericRouterConfigLifecycle(t *testing.T) {
	databasePath := filepath.Join(t.TempDir(), "router.sqlite3")
	config := PluginConfig{
		Version: 1,
		Components: []PluginComponentSpec{
			{
				Kind:    "router",
				Enabled: true,
				Config: map[string]any{
					"version":       1,
					"mode":          "shadow",
					"project_id":    "go-router-test",
					"database_path": databasePath,
					"pools": []any{
						map[string]any{
							"id":                        "go-router-pool",
							"api_family":                "openai_chat_completions",
							"anchor_models":             []string{"anchor-model"},
							"anchor_revision":           "2026-07-11",
							"sampling_probability":      1.0,
							"max_candidates_per_sample": 1,
							"selector":                  map[string]any{},
							"concurrency": map[string]any{
								"shadow":      1,
								"judge":       1,
								"max_pending": 2,
							},
							"candidates": []any{
								map[string]any{
									"id":             "candidate",
									"model":          "candidate-model",
									"model_revision": "2026-07-11",
									"cost_rank":      0,
								},
							},
							"judge": map[string]any{
								"version":                1,
								"model":                  "judge-model",
								"model_revision":         "2026-07-11",
								"prompt_version":         "pairwise-equivalence-v1",
								"rubric_version":         "response-trajectory-equivalence-v1",
								"output_schema_version":  1,
								"response_weight":        0.5,
								"trajectory_weight":      0.5,
								"response_floor":         0.8,
								"trajectory_floor":       0.8,
								"judge_confidence_floor": 0.7,
								"pass_threshold":         0.85,
								"max_rationale_bytes":    4096,
								"base_cooloff_seconds":   10,
								"max_cooloff_seconds":    300,
							},
						},
					},
				},
			},
		},
	}

	kinds, err := ListPluginKinds()
	if err != nil {
		t.Fatalf("ListPluginKinds failed: %v", err)
	}
	found := false
	for _, kind := range kinds {
		if kind == "router" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("router missing from generic plugin kinds: %#v", kinds)
	}

	validation, err := ValidatePluginConfig(config)
	if err != nil {
		t.Fatalf("ValidatePluginConfig failed: %v", err)
	}
	if len(validation.Diagnostics) != 0 {
		t.Fatalf("unexpected Router validation diagnostics: %#v", validation.Diagnostics)
	}

	defer func() {
		if err := ClearPluginConfiguration(); err != nil {
			t.Errorf("ClearPluginConfiguration failed: %v", err)
		}
	}()
	initialization, err := InitializePlugins(config)
	if err != nil {
		t.Fatalf("InitializePlugins failed: %v", err)
	}
	if len(initialization.Diagnostics) != 0 {
		t.Fatalf("unexpected Router initialization diagnostics: %#v", initialization.Diagnostics)
	}
	if _, err := os.Stat(databasePath); err != nil {
		t.Fatalf("Router database was not initialized: %v", err)
	}
	active, err := ActivePluginReport()
	if err != nil {
		t.Fatalf("ActivePluginReport failed: %v", err)
	}
	if active == nil || len(active.Diagnostics) != 0 {
		t.Fatalf("unexpected active Router report: %#v", active)
	}
}

func TestRouterRemainsGenericAndCallIneligibleInGo(t *testing.T) {
	if _, err := os.Stat("router"); !os.IsNotExist(err) {
		t.Fatalf("typed Go Router package must not exist, stat error: %v", err)
	}
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob Go sources: %v", err)
	}
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		contents, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		text := string(contents)
		for _, forbidden := range []string{"LlmReplayFactory", "LlmExecuteV2", "nemo_relay_router_"} {
			if strings.Contains(text, forbidden) {
				t.Fatalf("%s unexpectedly exposes %q", file, forbidden)
			}
		}
	}
}
